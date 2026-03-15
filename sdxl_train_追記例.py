# 文頭付近に下記を追加
from custom.loss_extra_calc import calc_extra_losses


# 中略


    loss_recorder = train_util.LossRecorder()
    for epoch in range(num_train_epochs):
        #省略
        
        for step, batch in enumerate(train_dataloader):
            current_step.value = global_step

            # 省略

                    noise, noisy_latents, timesteps, huber_c = train_util.get_noise_noisy_latents_and_timesteps(
                        args, noise_scheduler, latents
                    )

                    if (
                        args.min_snr_gamma
                        or args.scale_v_pred_loss_like_noise_pred
                        or args.v_pred_like_loss
                        or args.debiased_estimation_loss
                        or args.masked_loss
                    ):
                        # do not mean over batch dimension for snr weight or scale v-pred loss
                        loss = train_util.conditional_loss(
                            noise_pred.float(), target.float(), reduction="none", loss_type=args.loss_type, huber_c=huber_c
                        )

                        if args.masked_loss or ("alpha_masks" in batch and batch["alpha_masks"] is not None):
                            loss = apply_masked_loss(loss, batch)
                            #loss_base_raw = apply_masked_loss(loss_base_raw, batch)

                        # これを追加してください ==================================================
                        is_use_extra_loss = True
                        if is_use_extra_loss:

                            snr = noise_scheduler.all_snr[timesteps]
                            if args.v_parameterization:
                                snr_weight = 1.0 / (snr + 1.0)
                            else:
                                snr_weight = one / 1.0
                                
                            snr_weight = 1.0 - snr_weight # 反転処理とdtype変換
                                
                            # loss.dim() が 4 なら [B, 1, 1, 1]、3 なら [B, 1, 1]
                            snr_weight_view = snr_weight.view(snr_weight.shape[0], *([1] * (target.dim() - 1)))
                            
                            current_mask = batch["alpha_masks"] if (args.masked_loss and "alpha_masks" in batch) else None
                            
                            loss = calc_extra_losses(
                                loss, 
                                target, 
                                noise_pred, 
                                args, 
                                huber_c, 
                                global_step, 
                                accelerator,
                                snr_weight_view,
                                current_mask,
                            )
                        
                        # ========================================================================
                        
                        #以下は、sd-scriptsオリジナルのコード
                        loss = loss.mean([1, 2, 3])
                        
                        if args.min_snr_gamma:
                            loss = apply_snr_weight(loss, timesteps, noise_scheduler, args.min_snr_gamma, args.v_parameterization)
                        if args.scale_v_pred_loss_like_noise_pred:
                            loss = scale_v_prediction_loss_like_noise_prediction(loss, timesteps, noise_scheduler)
                        if args.v_pred_like_loss:
                            loss = add_v_prediction_like_loss(loss, timesteps, noise_scheduler, args.v_pred_like_loss)
                        if args.debiased_estimation_loss:
                            loss = apply_debiased_estimation(loss, timesteps, noise_scheduler, args.v_parameterization)

                        loss = loss.mean()  # mean over batch dimension
                    else:
                        loss = train_util.conditional_loss(
                            noise_pred.float(), target.float(), reduction="mean", loss_type=args.loss_type, huber_c=huber_c
                        )
                        
                        # 上述のmin_snr_gamma, v_predなどのオプションを使用しない場合は、
                        # ここに上述の追加内容を、ここへ追加してください