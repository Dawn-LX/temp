from easydict import EasyDict
import numpy as np
import torch

# a debug demo of Diffusion Forcing 
class Demo:
    def __init__(self,cfg) -> None:
        self.cfg = cfg
        self.sampling_timesteps = cfg.sampling_timesteps
        pass
        
        self.frame_stack = 1 # configurations/algorithm/df_base.yaml
        self.chunk_size = 1 # configurations/algorithm/df_base.yaml
        self.n_frames = 16 # for dmlab video dataset, configurations/dataset/video_dmlab.yaml
        self.context_frames = 2 # for dmlab video dataset, configurations/dataset/video_dmlab.yaml
        self.n_tokens = self.n_frames // self.frame_stack

        self.x_shape = (3,128,128) # refer to configurations/dataset/base_video.yaml
        self.x_stacked_shape = list(self.x_shape)
        self.x_stacked_shape[0] *= self.frame_stack


        self.clip_noise = 6.0 # configurations/algorithm/df_video.yaml
        self.device = torch.device("cpu")
        


    def _generate_scheduling_matrix(self, horizon: int):
        match self.cfg.scheduling_matrix:
            case "pyramid":
                return self._generate_pyramid_scheduling_matrix(horizon, self.uncertainty_scale)
            case "full_sequence":
                return np.arange(self.sampling_timesteps, -1, -1)[:, None].repeat(horizon, axis=1)
            case "autoregressive":
                '''
                # sampling_timesteps = T, horizon = L
                # shape  == (height, L), height = ((L-1)*T)+1 + T = L*T + 1
                currently, in practice, for video prediction horizon=L=1 : 
                    i.e., shape == (1,T+1), i.e., [T,T-1,T-2, ..., 1,0].transpose()
                
                in general, for scheduling_matrix of shape (L*T+1, L)
                e.g., L=4, T= 10, shape = (41,1)
                the matrix is like 
                [[10 10 10 10]
                 [ 9 10 10 10]
                 ...
                 [ 1 10 10 10]

                 [ 0 10 10 10]
                 [ 0  9 10 10]
                 [ 0  8 10 10]
                 ...
                 [ 0  1 10 10]

                 [ 0  0 10 10]
                 [ 0  0  9 10]
                 [ 0  0  8 10]
                 ...
                 [ 0  0  1 10]

                 [ 0  0  0 10]
                 [ 0  0  0  9]
                 ...
                 [ 0  0  0  1]

                 [ 0  0  0  0]]
                '''
                
                return self._generate_pyramid_scheduling_matrix(horizon, self.sampling_timesteps)
                
            case "trapezoid":
                return self._generate_trapezoid_scheduling_matrix(horizon, self.uncertainty_scale)

    def _generate_pyramid_scheduling_matrix(self, horizon: int, uncertainty_scale: float):
        height = self.sampling_timesteps + int((horizon - 1) * uncertainty_scale) + 1
        scheduling_matrix = np.zeros((height, horizon), dtype=np.int64)
        for m in range(height):
            for t in range(horizon):
                scheduling_matrix[m, t] = self.sampling_timesteps + int(t * uncertainty_scale) - m

        return np.clip(scheduling_matrix, 0, self.sampling_timesteps)
    
    @torch.no_grad()
    def validation_step(self):
        # xs, conditions, masks = self._preprocess_batch(batch)
        batch_size = 1
        xs = torch.randn(size=(self.n_frames,batch_size,*self.x_shape))

        n_frames, batch_size, *_ = xs.shape
        xs_pred = []
        curr_frame = 0

        # context
        n_context_frames = self.context_frames // self.frame_stack
        xs_pred = xs[:n_context_frames].clone()
        curr_frame += n_context_frames

        # pbar = tqdm(total=n_frames, initial=curr_frame, desc="Sampling")
        while curr_frame < n_frames:
            if self.chunk_size > 0:
                horizon = min(n_frames - curr_frame, self.chunk_size) # currently,  `chunk_size==1` for video prediction
            else:
                horizon = n_frames - curr_frame
            assert horizon <= self.n_tokens, "horizon exceeds the number of tokens."
            scheduling_matrix = self._generate_scheduling_matrix(horizon)

            chunk = torch.randn((horizon, batch_size, *self.x_stacked_shape), device=self.device)
            chunk = torch.clamp(chunk, -self.clip_noise, self.clip_noise)
            xs_pred = torch.cat([xs_pred, chunk], 0)

            # sliding window: only input the last n_tokens frames
            start_frame = max(0, curr_frame + horizon - self.n_tokens)
            # self.n_tokens = cfg.n_frames // cfg.frame_stack  # for video prediction, currently `frame_stack==1`

            # pbar.set_postfix(
            #     {
            #         "start": start_frame,
            #         "end": curr_frame + horizon,
            #     }
            # )
            print(f"curr_frame={curr_frame},horizon={horizon}")
            print("   ", {"start_frame": start_frame,"end_frame=curr_frame + horizon": curr_frame + horizon})
            print("   ", f"xs_pred.shape={xs_pred.shape}; xs_pred[start_frame:].shape={xs_pred[start_frame:].shape}")
            # print("   ", "scheduling_matrix:",scheduling_matrix,scheduling_matrix.shape)
            for m in range(scheduling_matrix.shape[0] - 1): # loop for noisy level
                from_noise_levels = np.concatenate((np.zeros((curr_frame,), dtype=np.int64), scheduling_matrix[m]))[
                    :, None
                ].repeat(batch_size, axis=1)
                to_noise_levels = np.concatenate(
                    (
                        np.zeros((curr_frame,), dtype=np.int64),
                        scheduling_matrix[m + 1],
                    )
                )[
                    :, None
                ].repeat(batch_size, axis=1)

                from_noise_levels = torch.from_numpy(from_noise_levels).to(self.device)
                to_noise_levels = torch.from_numpy(to_noise_levels).to(self.device)  # (n_frames, batch_size)
                
                print("   ", f"from_noise_levels[:,0] :{from_noise_levels[:,0].tolist()}",f"  to_noise_levels[:,0] :{to_noise_levels[:,0].tolist()}")


                # update xs_pred by DDIM or DDPM sampling
                # input frames within the sliding window
                continue
                xs_pred[start_frame:] = self.diffusion_model.sample_step(
                    xs_pred[start_frame:],
                    conditions[start_frame : curr_frame + horizon],
                    from_noise_levels[start_frame:],
                    to_noise_levels[start_frame:],
                )

            curr_frame += horizon
            

        return xs,xs_pred
    
        # FIXME: loss
        loss = F.mse_loss(xs_pred, xs, reduction="none")
        loss = self.reweight_loss(loss, masks)

        xs = self._unstack_and_unnormalize(xs)
        xs_pred = self._unstack_and_unnormalize(xs_pred)
        self.validation_step_outputs.append((xs_pred.detach().cpu(), xs.detach().cpu()))

        return loss

def demo_scheduling_matrix():
    config = EasyDict(
        scheduling_matrix = "autoregressive",
        sampling_timesteps = 10,
    )
    demo = Demo(config)

    
    scheduling_matrix_h1 = demo._generate_scheduling_matrix(1)
    scheduling_matrix_h10 = demo._generate_scheduling_matrix(4)

    print(scheduling_matrix_h1,scheduling_matrix_h1.shape)
    print(scheduling_matrix_h10,scheduling_matrix_h10.shape)

if __name__ == "__main__":
    config = EasyDict(
        scheduling_matrix = "autoregressive",
        sampling_timesteps = 10,
    )
    demo = Demo(config)

    demo.validation_step()

