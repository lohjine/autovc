from model_vc import Generator
import torch
import torch.nn.functional as F
import time
import datetime

class Solver(object):

    def __init__(self, vcc_loader, config):
        """Initialize configurations."""

        # Data loader.
        self.vcc_loader = vcc_loader

        # Model configurations.
        self.lambda_cd = config.lambda_cd
        self.dim_neck = config.dim_neck
        self.dim_emb = config.dim_emb
        self.dim_pre = config.dim_pre
        self.freq = config.freq

        # Training configurations.
        self.batch_size = config.batch_size
        self.num_iters = config.num_iters
        self.start_iter = 0

        # Miscellaneous.
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device('cuda:0' if self.use_cuda else 'cpu')
        self.log_step = config.log_step
        self.checkpoint = config.checkpoint
        self.resume = config.resume
        self.running_avg_mean = 0
        self.running_avg_sd = 0

        # Build the model and tensorboard.
        self.build_model()


    def build_model(self):

        self.G = Generator(self.dim_neck, self.dim_emb, self.dim_pre, self.freq)

        self.g_optimizer = torch.optim.Adam(self.G.parameters(), 0.0001)

        ## TODO: load checkpoint here

        # this is from make_metadata.py
        #c_checkpoint = torch.load('3000000-BL.ckpt')
        #new_state_dict = OrderedDict()
        #for key, val in c_checkpoint['model_b'].items():
        #    new_key = key[7:]
        #    new_state_dict[new_key] = val
        #C.load_state_dict(new_state_dict)

        # this is from waveglow
        # model.load_state_dict( checkpoint_dict['model'].state_dict() )
        # self.start_iter = checkpoint_dict['iteration']
        # self.g_optimizer.load_state_dict(checkpoint_dict['optimizer'])

        # this is from conversion.ipynb
        #g_checkpoint = torch.load('autovc.ckpt')
        #G.load_state_dict(g_checkpoint['model'])


        self.G.to(self.device)
        
        if self.resume:
            g_checkpoint = torch.load(self.resume)
            self.G.load_state_dict(g_checkpoint['model'])
            self.g_optimizer.load_state_dict(g_checkpoint['optimizer'])
            self.start_iter = g_checkpoint['iteration']


    def reset_grad(self):
        """Reset the gradient buffers."""
        self.g_optimizer.zero_grad()


    #=====================================================================================================================================#



    def train(self):
        # Set data loader.
        data_loader = self.vcc_loader

        # Print logs in specified order
        keys = ['G/loss_id','G/loss_id_psnt','G/loss_cd']

        # Start training.
        print('Start training...')
        start_time = time.time()
        for i in range(self.start_iter, self.num_iters):

            # =================================================================================== #
            #                             1. Preprocess input data                                #
            # =================================================================================== #

            # Fetch data.
            try:
                x_real, emb_org = next(data_iter)
            except:
                data_iter = iter(data_loader)
                x_real, emb_org = next(data_iter)


            x_real = x_real.to(self.device)
            emb_org = emb_org.to(self.device)


            # =================================================================================== #
            #                               2. Train the generator                                #
            # =================================================================================== #

            self.G = self.G.train()

            # Identity mapping loss
            x_identic, x_identic_psnt, code_real = self.G(x_real, emb_org, emb_org)
            g_loss_id = F.mse_loss(x_real, x_identic)
            g_loss_id_psnt = F.mse_loss(x_real, x_identic_psnt)

            # Code semantic loss.
            code_reconst = self.G(x_identic_psnt, emb_org, None)
            g_loss_cd = F.l1_loss(code_real, code_reconst)


            # Backward and optimize.
            g_loss = g_loss_id + g_loss_id_psnt + self.lambda_cd * g_loss_cd
            self.reset_grad()
            g_loss.backward()
            self.g_optimizer.step()

            # Logging.
            loss = {}
            loss['G/loss_id'] = g_loss_id.item()
            loss['G/loss_id_psnt'] = g_loss_id_psnt.item()
            loss['G/loss_cd'] = g_loss_cd.item()

            # =================================================================================== #
            #                                 4. Miscellaneous                                    #
            # =================================================================================== #

            if self.running_avg_mean == 0:
                self.running_avg_mean = loss['G/loss_cd']
            else:
                self.running_avg_mean = self.running_avg_mean + ((loss['G/loss_cd'] - self.running_avg_mean) / 100)
                
            # Print out training information.
            if (i+1) % self.log_step == 0:
                et = time.time() - start_time
                et = str(datetime.timedelta(seconds=et))[:-7]
                log = "Elapsed [{}], Iteration [{}/{}]".format(et, i+1, self.num_iters)
                for tag in keys:
                    log += ", {}: {:.4f}".format(tag, loss[tag])
                    
                log += f", mvg_avg:{'%.04f'%self.running_avg}+-{}"
                
                print(log)


            # Save checkpoint
            if i>0 and (i % self.checkpoint) == 0:
#                model_for_saving.load_state_dict()
                torch.save({'model': self.G.state_dict(),
                            'iteration': i,
                            'optimizer': self.g_optimizer.state_dict()}, f'checkpoint/chkpt_{i}')


