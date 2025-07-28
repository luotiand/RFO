# （省略重复的导入和函数定义）

def main(config):
    # （省略设备设置、数据加载等重复代码）

    if train:
        opt = optim.Adam(params=score_net.parameters(), lr=lr, betas=(0.5, 0.999))
        criterion = nn.MSELoss()

        training_loss = [None for _ in range(niter)]
        mape_records = []
        mape_iterations = []
        
        # --------------------------
        # 核心修改：跟踪最低训练损失
        # --------------------------
        best_loss = float('inf')  # 初始化为无穷大
        best_model_path = f"{para_path}best_loss_{model_name}"  # 最优损失模型保存路径
        
        start_time = time.time()
        iter_start_time = start_time
        check_interval = max(1, niter // 10)

        for it in range(niter):
            # 训练循环（每个epoch遍历所有batch）
            epoch_loss = []  # 记录当前epoch的所有batch损失
            for a_batch, x_batch in dataloader:
                a_ = (a_batch.to(device).to(dtype=torch.float64) - a_mean) / a_std
                x_ = (x_batch.to(device).to(dtype=torch.float64) - x_mean) / x_std
                
                t = torch.rand(batch_size, 1, device=device).to(dtype=torch.float64)
                t = t.view(batch_size, *([1] * (len(a_.shape) - 1)))
                opt.zero_grad()
                
                xt_ = rf.straight_process(a_, x_, t)
                exact_score = x_ - a_
                score = score_net(xt_, t)
                loss = criterion(exact_score, score)

                loss.backward()
                opt.step()
                
                epoch_loss.append(loss.item())  # 记录当前batch的损失
                
                # 释放显存
                del a_, x_, t, xt_, exact_score, score
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            
            # 当前epoch的平均损失（或最后一个batch的损失，根据需求选择）
            current_loss = np.mean(epoch_loss)  # 推荐：取epoch内所有batch的平均损失
            # current_loss = loss.item()  # 可选：仅用最后一个batch的损失
            training_loss[it] = current_loss

            # --------------------------
            # 核心修改：更新并保存最低损失模型
            # --------------------------
            if current_loss < best_loss:
                best_loss = current_loss
                torch.save(score_net.state_dict(), best_model_path)
                logging.info(f"更新最优损失模型 (Loss: {best_loss:.8f})，保存至 {best_model_path}")

            # 记录MAPE（仅作为监控，不影响模型保存）
            if (it + 1) % check_interval == 0 or it == niter - 1:
                score_net.eval()
                with torch.no_grad():
                    # （省略前向预测代码）
                    current_mape = calculate_mape(xt_last, x_true)
                    mape_records.append(current_mape)
                    mape_iterations.append(it + 1)
                    logging.info(f"Iteration {it + 1}/{niter}, Loss: {current_loss:.8f}, MAPE: {current_mape:.4f}%")
                score_net.train()

            # 学习率衰减（保持不变）
            if (it+1) % (niter//4) == 0:
                new_lr = opt.param_groups[0]['lr'] / 4
                logging.info(f"LR reduced from {opt.param_groups[0]['lr']} to {new_lr}")
                opt.param_groups[0]['lr'] = new_lr

            # 进度日志（保持不变）
            if (it + 1) % 5 == 0:
                # （省略进度计算代码）
                logging.info(f"Iter {it+1}/{niter}, Loss: {current_loss:.8f}")

        # 保存最终模型（同时保留最优损失模型）
        final_model_path = f"{para_path}{model_name}"
        torch.save(score_net.state_dict(), final_model_path)
        logging.info(f"训练结束。最终模型保存至 {final_model_path}，最优损失模型保存至 {best_model_path} (Best Loss: {best_loss:.8f})")

        # 绘制训练曲线（添加最优损失标注）
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        ax1.plot(range(1, niter+1), training_loss, 'b-')
        ax1.axhline(y=best_loss, color='g', linestyle='--', label=f'Best Loss: {best_loss:.8f}')  # 标注最低损失
        ax1.set_ylabel("Training Loss", fontsize=14)
        ax1.set_yscale("log")
        ax1.set_title("Training Loss Curve", fontsize=16)
        ax1.grid(True)
        ax1.legend()
        
        ax2.plot(mape_iterations, mape_records, 'r-o', label='Relative Error')
        ax2.set_xlabel("Iteration", fontsize=14)
        ax2.set_ylabel("MAPE (%)", fontsize=14)
        ax2.set_title("Validation MAPE", fontsize=16)
        ax2.grid(True)
        ax2.legend()
        
        plt.savefig(f"{save_path}{scorenet_model_class.lower()}_{target_size}_metrics.png", dpi=300)
        plt.close()

    # 评估模型（优先加载最优损失模型）
    best_model_path = f"{para_path}best_loss_{model_name}"
    if os.path.exists(best_model_path):
        state_dict = torch.load(best_model_path, map_location=device)
        logging.info(f"加载最优损失模型进行评估: {best_model_path} (Best Loss: {best_loss:.8f})")
    else:
        state_dict = torch.load(f"{para_path}{model_name}", map_location=device)
        logging.info(f"未找到最优损失模型，加载最终模型进行评估: {model_name}")
    
    # （省略后续评估和绘图代码）