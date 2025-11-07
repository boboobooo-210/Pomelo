import torch
import torch.nn as nn
import os
import sys
import json

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools import builder
from utils import misc, dist_utils
import time
from utils.logger import *
from utils.AverageMeter import AverageMeter
from utils.metrics import Metrics
from extensions.chamfer_dist import ChamferDistanceL1, ChamferDistanceL2
import math
import cv2
import numpy as np

def compute_loss(loss_1, loss_2, config, niter, train_writer):
    '''
    compute the final loss for optimization
    For dVAE: loss_1 : reconstruction loss, loss_2 : kld loss
    '''
    start = config.kldweight.start
    target = config.kldweight.target
    ntime = config.kldweight.ntime

    _niter = niter - 10000
    if _niter > ntime:
        kld_weight = target
    elif _niter < 0:
        kld_weight = 0.
    else:
        kld_weight = target + (start - target) *  (1. + math.cos(math.pi * float(_niter) / ntime)) / 2.

    if train_writer is not None:
        train_writer.add_scalar('Loss/Batch/KLD_Weight', kld_weight, niter)

    loss = loss_1 + kld_weight * loss_2

    return loss

def get_temp(config, niter):
    if config.get('temp') is not None:
        start = config.temp.start
        target = config.temp.target
        ntime = config.temp.ntime
        if niter > ntime:
            return target
        else:
            temp = target + (start - target) *  (1. + math.cos(math.pi * float(niter) / ntime)) / 2.
            return temp
    else:
        return 0 

def run_net(args, config, train_writer=None, val_writer=None):
    logger = get_logger(args.log_name)
    # build dataset
    (train_sampler, train_dataloader), (_, test_dataloader) = builder.dataset_builder(args, config.dataset.train), \
                                                            builder.dataset_builder(args, config.dataset.val)
    # build model
    base_model = builder.model_builder(config.model) # todo 也是相同的手法找到并构造模型
    if args.use_gpu:
        base_model.to(args.local_rank)

    # parameter setting
    start_epoch = 0
    best_metrics = None
    metrics = None

    # resume ckpts
    if args.resume:
        start_epoch, best_metrics = builder.resume_model(base_model, args, logger = logger)
        best_metrics = Metrics(config.consider_metric, best_metrics)
    elif args.start_ckpts is not None:
        builder.load_model(base_model, args.start_ckpts, logger = logger)

    # DDP
    if args.distributed:
        # Sync BN
        if args.sync_bn:
            base_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(base_model)
            print_log('Using Synchronized BatchNorm ...', logger = logger)
        base_model = nn.parallel.DistributedDataParallel(base_model, device_ids=[args.local_rank % torch.cuda.device_count()])
        print_log('Using Distributed Data parallel ...' , logger = logger)
    else:
        print_log('Using Data parallel ...' , logger = logger)
        base_model = nn.DataParallel(base_model).cuda()
    # optimizer & scheduler
    optimizer, scheduler = builder.build_opti_sche(base_model, config)
    
    # Criterion
    ChamferDisL1 = ChamferDistanceL1()
    ChamferDisL2 = ChamferDistanceL2()


    if args.resume:
        builder.resume_optimizer(optimizer, args, logger = logger)

    # trainval
    # training
    base_model.zero_grad()
    for epoch in range(start_epoch, config.max_epoch + 1):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        base_model.train()

        epoch_start_time = time.time()
        batch_start_time = time.time()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter(['Loss1', 'Loss2'])

        num_iter = 0

        base_model.train()  # set model to training mode
        n_batches = len(train_dataloader)
        for idx, (taxonomy_ids, model_ids, data) in enumerate(train_dataloader):
            num_iter += 1
            n_itr = epoch * n_batches + idx
            
            data_time.update(time.time() - batch_start_time)
            npoints = config.dataset.train._base_.N_POINTS
            dataset_name = config.dataset.train._base_.NAME
            if dataset_name == 'ShapeNet':
                points = data.cuda()
            elif dataset_name == 'MARS':
                points = data.cuda()
            elif dataset_name == 'MMFI':
                points = data.cuda()
            elif dataset_name == 'NTUAugmented':
                points = data.cuda()
            elif dataset_name == 'NTU':
                points = data.cuda()
            elif dataset_name == 'NTUPredH5Dataset':
                points = data.cuda()
            elif dataset_name == 'NTU_Skeleton_Raw':
                points = data.cuda()
            else:
                raise NotImplementedError(f'Train phase do not support {dataset_name}')

            temp = get_temp(config, n_itr)


            ret = base_model(points, temperature = temp, hard = False)

            # SkeletonTokenizer现在使用原始dVAE的损失计算方式
            loss_1, loss_2 = base_model.module.get_loss(ret, points)
            _loss = compute_loss(loss_1, loss_2, config, n_itr, train_writer)

            _loss.backward()

            # forward
            if num_iter == config.step_per_update:
                num_iter = 0
                optimizer.step()
                base_model.zero_grad()

            if args.distributed:
                loss_1 = dist_utils.reduce_tensor(loss_1, args)
                loss_2 = dist_utils.reduce_tensor(loss_2, args)
                losses.update([loss_1.item() * 1000, loss_2.item() * 1000])
            else:
                losses.update([loss_1.item() * 1000, loss_2.item() * 1000])


            if args.distributed:
                torch.cuda.synchronize()


            if train_writer is not None:
                train_writer.add_scalar('Loss/Batch/Loss_1', loss_1.item() * 1000, n_itr)
                train_writer.add_scalar('Loss/Batch/Loss_2', loss_2.item() * 1000, n_itr)
                train_writer.add_scalar('Loss/Batch/Temperature', temp, n_itr)
                train_writer.add_scalar('Loss/Batch/LR', optimizer.param_groups[0]['lr'], n_itr)


            batch_time.update(time.time() - batch_start_time)
            batch_start_time = time.time()

            if idx % 20 == 0:
                print_log('[Epoch %d/%d][Batch %d/%d] BatchTime = %.3f (s) DataTime = %.3f (s) Losses = %s lr = %.6f' %
                            (epoch, config.max_epoch, idx + 1, n_batches, batch_time.val(), data_time.val(),
                            ['%.4f' % l for l in losses.val()], optimizer.param_groups[0]['lr']), logger = logger)
        if config.scheduler.type != 'function':
            if isinstance(scheduler, list):
                for item in scheduler:
                    item.step(epoch)
            else:
                scheduler.step(epoch)
        epoch_end_time = time.time()

        if train_writer is not None:
            train_writer.add_scalar('Loss/Epoch/Loss_1', losses.avg(0), epoch)
            train_writer.add_scalar('Loss/Epoch/Loss_2', losses.avg(1), epoch)

        print_log('[Training] EPOCH: %d EpochTime = %.3f (s) Losses = %s' %
            (epoch,  epoch_end_time - epoch_start_time, ['%.4f' % l for l in losses.avg()]), logger = logger)

        if epoch % args.val_freq == 0 and epoch != 0:
            # Validate the current model
            metrics = validate(base_model, test_dataloader, epoch, ChamferDisL1, ChamferDisL2, val_writer, args, config, logger=logger)

            # Save ckeckpoints
            if  metrics.better_than(best_metrics):
                best_metrics = metrics
                builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, 'ckpt-best', args, logger = logger)
        builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, 'ckpt-last', args, logger = logger)      
        if (config.max_epoch - epoch) < 5:
            builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, f'ckpt-epoch-{epoch:03d}', args, logger = logger)   
    if train_writer is not None:  
        train_writer.close()
    if val_writer is not None:
        val_writer.close()

def validate(base_model, test_dataloader, epoch, ChamferDisL1, ChamferDisL2, val_writer, args, config, logger = None):
    print_log(f"[VALIDATION] Start validating epoch {epoch}", logger = logger)
    base_model.eval()  # set model to eval mode

    test_losses = AverageMeter(['SparseLossL1', 'SparseLossL2', 'DenseLossL1', 'DenseLossL2'])
    test_metrics = AverageMeter(Metrics.names())
    category_metrics = dict()
    n_samples = len(test_dataloader) # bs is 1

    with torch.no_grad():
        for idx, (taxonomy_ids, model_ids, data) in enumerate(test_dataloader):
            taxonomy_id = taxonomy_ids[0] if isinstance(taxonomy_ids[0], str) else taxonomy_ids[0].item()
            model_id = model_ids[0]

            npoints = config.dataset.val._base_.N_POINTS
            dataset_name = config.dataset.val._base_.NAME
            if dataset_name == 'ShapeNet':
                points = data.cuda()
            elif dataset_name == 'MARS':
                points = data.cuda()
            elif dataset_name == 'MMFI':
                points = data.cuda()
            elif dataset_name == 'NTUAugmented':
                points = data.cuda()
            elif dataset_name == 'NTU':
                points = data.cuda()
            elif dataset_name == 'NTUPredH5Dataset':
                points = data.cuda()
            elif dataset_name == 'NTU_Skeleton_Raw':
                points = data.cuda()
            else:
                raise NotImplementedError(f'Validation phase do not support {dataset_name}')

            ret = base_model(inp = points, hard=True, eval=True)
            coarse_points = ret[0]
            dense_points = ret[1]

            sparse_loss_l1 =  ChamferDisL1(coarse_points, points)
            sparse_loss_l2 =  ChamferDisL2(coarse_points, points)
            dense_loss_l1 =  ChamferDisL1(dense_points, points)
            dense_loss_l2 =  ChamferDisL2(dense_points, points)

            if args.distributed:
                sparse_loss_l1 = dist_utils.reduce_tensor(sparse_loss_l1, args)
                sparse_loss_l2 = dist_utils.reduce_tensor(sparse_loss_l2, args)
                dense_loss_l1 = dist_utils.reduce_tensor(dense_loss_l1, args)
                dense_loss_l2 = dist_utils.reduce_tensor(dense_loss_l2, args)

            test_losses.update([sparse_loss_l1.item() * 1000, sparse_loss_l2.item() * 1000, dense_loss_l1.item() * 1000, dense_loss_l2.item() * 1000])

            _metrics = Metrics.get(dense_points, points)

            if taxonomy_id not in category_metrics:
                category_metrics[taxonomy_id] = AverageMeter(Metrics.names())
            category_metrics[taxonomy_id].update(_metrics)

            vis_list = [0, 1000, 1600, 1800, 2400, 3400]
            if val_writer is not None and idx in vis_list: #% 200 == 0:
                input_pc = points.squeeze().detach().cpu().numpy()
                input_pc = misc.get_ptcloud_img(input_pc)
                val_writer.add_image('Model%02d/Input'% idx , input_pc, epoch, dataformats='HWC')

                sparse = coarse_points.squeeze().cpu().numpy()
                sparse_img = misc.get_ptcloud_img(sparse)
                val_writer.add_image('Model%02d/Sparse' % idx, sparse_img, epoch, dataformats='HWC')

                dense = dense_points.squeeze().cpu().numpy()
                dense_img = misc.get_ptcloud_img(dense)
                val_writer.add_image('Model%02d/Dense' % idx, dense_img, epoch, dataformats='HWC')
       
        
            if (idx+1) % 2000 == 0:
                print_log('Test[%d/%d] Taxonomy = %s Sample = %s Losses = %s Metrics = %s' %
                            (idx + 1, n_samples, taxonomy_id, model_id, ['%.4f' % l for l in test_losses.val()], 
                            ['%.4f' % m for m in _metrics]), logger=logger)
        for _,v in category_metrics.items():
            test_metrics.update(v.avg())
        print_log('[Validation] EPOCH: %d  Metrics = %s' % (epoch, ['%.4f' % m for m in test_metrics.avg()]), logger=logger)

        if args.distributed:
            torch.cuda.synchronize()
     
    # Print testing results
    dataset_name = config.dataset.val._base_.NAME
    if dataset_name == 'ShapeNet':
        shapenet_dict = json.load(open('./data/shapenet_synset_dict.json', 'r'))
    elif dataset_name == 'MARS':
        # MARS数据集使用简单的映射
        shapenet_dict = {'human_skeleton': 'Human Skeleton'}
    elif dataset_name == 'NTUAugmented':
        # NTU数据集使用简单的映射
        shapenet_dict = {'human_skeleton': 'Human Skeleton'}
    elif dataset_name == 'NTU':
        # NTU数据集使用简单的映射
        shapenet_dict = {'human_skeleton': 'Human Skeleton'}
    elif dataset_name == 'NTUPredH5Dataset':
        # NTU-Pred H5数据集使用简单的映射
        shapenet_dict = {'human_skeleton': 'Human Skeleton'}
    elif dataset_name == 'NTU_Skeleton_Raw':
        # NTU原始骨架数据集使用简单的映射
        shapenet_dict = {'ntu_skeleton': 'NTU Skeleton'}
    else:
        shapenet_dict = {}
    
    print_log('============================ TEST RESULTS ============================',logger=logger)
    msg = ''
    msg += 'Taxonomy\t'
    msg += '#Sample\t'
    for metric in test_metrics.items:
        msg += metric + '\t'
    msg += '#ModelName\t'
    print_log(msg, logger=logger)

    for taxonomy_id in category_metrics:
        msg = ''
        msg += (taxonomy_id + '\t')
        msg += (str(category_metrics[taxonomy_id].count(0)) + '\t')
        for value in category_metrics[taxonomy_id].avg():
            msg += '%.3f \t' % value
        msg += shapenet_dict.get(taxonomy_id, taxonomy_id) + '\t'
        print_log(msg, logger=logger)

    msg = ''
    msg += 'Overall\t\t'
    for value in test_metrics.avg():
        msg += '%.3f \t' % value
    print_log(msg, logger=logger)

    # Add testing results to TensorBoard
    if val_writer is not None:
        val_writer.add_scalar('Loss/Epoch/Sparse', test_losses.avg(0), epoch)
        val_writer.add_scalar('Loss/Epoch/Dense', test_losses.avg(2), epoch)
        for i, metric in enumerate(test_metrics.items):
            val_writer.add_scalar('Metric/%s' % metric, test_metrics.avg(i), epoch)

    return Metrics(config.consider_metric, test_metrics.avg())

def test_net(args, config):
    logger = get_logger(args.log_name)
    print_log('Tester start ... ', logger = logger)
    _, test_dataloader = builder.dataset_builder(args, config.dataset.test)
 
    base_model = builder.model_builder(config.model)
    builder.load_model(base_model, args.ckpts, logger = logger)

    if args.use_gpu:
        base_model.to(args.local_rank)

    #  DDP    
    if args.distributed:
        raise NotImplementedError()

    test(base_model, test_dataloader, args, config, logger=logger)

def test(base_model, test_dataloader, args, config, logger = None):

    base_model.eval()  # set model to eval mode
    target = './vis'
    
    # 根据数据集类型设置有用的类别
    dataset_name = config.dataset.test._base_.NAME
    if dataset_name == 'ShapeNet':
        useful_cate = [
            "02691156", "02818832", "04379243", "04099429", "03948459", "03790512",
            "03642806", "03467517", "03261776", "03001627", "02958343", "03759954"
        ]
    elif dataset_name == 'MARS':
        useful_cate = ['human_skeleton']  # MARS只有人体骨架一个类别
    elif dataset_name == 'NTUAugmented':
        useful_cate = ['human_skeleton']  # NTU只有人体骨架一个类别
    elif dataset_name == 'NTU':
        useful_cate = ['human_skeleton']  # NTU只有人体骨架一个类别
    elif dataset_name == 'NTUPredH5Dataset':
        useful_cate = ['human_skeleton']  # NTU-Pred H5只有人体骨架一个类别
    elif dataset_name == 'NTU_Skeleton_Raw':
        useful_cate = ['ntu_skeleton']  # NTU原始骨架只有一个类别
        useful_cate = ['human_skeleton']  # NTU-Pred H5只有人体骨架一个类别
    else:
        useful_cate = []
    
    with torch.no_grad():
        for idx, (taxonomy_ids, model_ids, data) in enumerate(test_dataloader):
            # import pdb; pdb.set_trace()
            if useful_cate and taxonomy_ids[0] not in useful_cate:
                continue
    
            dataset_name = config.dataset.test._base_.NAME
            if dataset_name == 'ShapeNet':
                points = data.cuda()
            elif dataset_name == 'MARS':
                points = data.cuda()
            elif dataset_name == 'NTUAugmented':
                points = data.cuda()
            elif dataset_name == 'NTU':
                points = data.cuda()
            elif dataset_name == 'NTUPredH5Dataset':
                points = data.cuda()
            elif dataset_name == 'NTU_Skeleton_Raw':
                points = data.cuda()
            else:
                raise NotImplementedError(f'Test phase do not support {dataset_name}')


            ret = base_model(inp = points, hard=True, eval=True)
            dense_points = ret[1]

            final_image = []

            data_path = f'./vis/{taxonomy_ids[0]}_{idx}'
            if not os.path.exists(data_path):
                os.makedirs(data_path)

            points = points.squeeze().detach().cpu().numpy()
            np.savetxt(os.path.join(data_path,'gt.txt'), points, delimiter=';')
            points = misc.get_ptcloud_img(points)
            final_image.append(points)

            dense_points = dense_points.squeeze().detach().cpu().numpy()
            np.savetxt(os.path.join(data_path,'dense_points.txt'), dense_points, delimiter=';')
            dense_points = misc.get_ptcloud_img(dense_points)
            final_image.append(dense_points)

            img = np.concatenate(final_image, axis=1)
            img_path = os.path.join(data_path, f'plot.jpg')
            cv2.imwrite(img_path, img)

            if idx > 1000:
                break

        return

def visualize_mars_reconstruction(args, config, num_samples=10):
    """
    专门用于MARS数据集重建效果可视化的函数
    """
    logger = get_logger(args.log_name)
    print_log('Starting MARS reconstruction visualization...', logger=logger)
    
    # 构建数据集和模型
    _, test_dataloader = builder.dataset_builder(args, config.dataset.test)
    base_model = builder.model_builder(config.model)
    builder.load_model(base_model, args.ckpts, logger=logger)
    
    if args.use_gpu:
        base_model.to(args.local_rank)
    
    base_model.eval()
    
    # 创建可视化目录
    vis_dir = './vis/MARS_reconstruction'
    if not os.path.exists(vis_dir):
        os.makedirs(vis_dir)
    
    # Chamfer Distance 计算
    ChamferDisL1 = ChamferDistanceL1()
    ChamferDisL2 = ChamferDistanceL2()
    
    reconstruction_metrics = []
    
    with torch.no_grad():
        for idx, (taxonomy_ids, model_ids, data) in enumerate(test_dataloader):
            if idx >= num_samples:
                break
                
            points = data.cuda()
            
            # 前向传播
            ret = base_model(inp=points, hard=True, eval=True)
            coarse_points = ret[0]
            dense_points = ret[1]
            
            # 计算重建误差
            sparse_loss_l1 = ChamferDisL1(coarse_points, points)
            dense_loss_l1 = ChamferDisL1(dense_points, points)
            
            reconstruction_metrics.append({
                'sample_id': idx,
                'sparse_cd': sparse_loss_l1.item() * 1000,
                'dense_cd': dense_loss_l1.item() * 1000
            })
            
            # 转换为numpy数组
            original = points.squeeze().detach().cpu().numpy()
            sparse = coarse_points.squeeze().detach().cpu().numpy()  
            dense = dense_points.squeeze().detach().cpu().numpy()
            
            # 保存点云数据
            sample_dir = os.path.join(vis_dir, f'sample_{idx:03d}')
            os.makedirs(sample_dir, exist_ok=True)
            
            np.savetxt(os.path.join(sample_dir, 'original.txt'), original, delimiter=',')
            np.savetxt(os.path.join(sample_dir, 'sparse_recon.txt'), sparse, delimiter=',')
            np.savetxt(os.path.join(sample_dir, 'dense_recon.txt'), dense, delimiter=',')
            
            # 生成可视化图像
            final_image = []
            
            # 原始点云图像
            original_img = misc.get_ptcloud_img(original)
            final_image.append(original_img)
            
            # 粗糙重建图像
            sparse_img = misc.get_ptcloud_img(sparse)
            final_image.append(sparse_img)
            
            # 精细重建图像
            dense_img = misc.get_ptcloud_img(dense)
            final_image.append(dense_img)
            
            # 水平拼接图像
            combined_img = np.concatenate(final_image, axis=1)
            
            # 添加标题
            cv2.putText(combined_img, f'Sample {idx} - Original | Sparse | Dense', 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(combined_img, f'Sparse CD: {sparse_loss_l1.item()*1000:.3f}', 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(combined_img, f'Dense CD: {dense_loss_l1.item()*1000:.3f}', 
                       (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # 保存图像
            img_path = os.path.join(sample_dir, 'reconstruction_comparison.jpg')
            cv2.imwrite(img_path, combined_img)
            
            print_log(f'Sample {idx}: Sparse CD = {sparse_loss_l1.item()*1000:.3f}, '
                     f'Dense CD = {dense_loss_l1.item()*1000:.3f}', logger=logger)
    
    # 计算总体统计
    avg_sparse_cd = np.mean([m['sparse_cd'] for m in reconstruction_metrics])
    avg_dense_cd = np.mean([m['dense_cd'] for m in reconstruction_metrics])
    
    print_log('============================ RECONSTRUCTION SUMMARY ============================', logger=logger)
    print_log(f'Average Sparse Chamfer Distance: {avg_sparse_cd:.3f}', logger=logger)
    print_log(f'Average Dense Chamfer Distance: {avg_dense_cd:.3f}', logger=logger)
    print_log(f'Reconstruction Improvement: {((avg_sparse_cd - avg_dense_cd) / avg_sparse_cd * 100):.1f}%', logger=logger)
    print_log(f'Visualizations saved to: {vis_dir}', logger=logger)
    
    # 保存统计数据
    import json
    stats = {
        'average_sparse_cd': avg_sparse_cd,
        'average_dense_cd': avg_dense_cd,
        'improvement_percentage': (avg_sparse_cd - avg_dense_cd) / avg_sparse_cd * 100,
        'sample_metrics': reconstruction_metrics
    }
    
    with open(os.path.join(vis_dir, 'reconstruction_stats.json'), 'w') as f:
        json.dump(stats, f, indent=2)
    
    return stats
