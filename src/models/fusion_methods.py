import torch
from torch import nn
from torch.nn import functional as F
from .fusion_utils import GanGenerator, GanDiscriminator


class AutoFusion(nn.Module):
    """docstring for AutoFusion"""
    def __init__(self, config, input_features):
        super(AutoFusion, self).__init__()
        self.config = config
        self.input_features = input_features

        self.fuse_in = nn.Sequential(
            nn.Linear(input_features, input_features//2),
            nn.Tanh(),
            nn.Linear(input_features//2, config['latent_dim']),
            nn.ReLU()
            )
        self.fuse_out = nn.Sequential(
            nn.Linear(config['latent_dim'], input_features//2),
            nn.ReLU(),
            nn.Linear(input_features//2, input_features)
            )
        self.criterion = nn.MSELoss()

    def forward(self, z):
        compressed_z = self.fuse_in(z)
        loss = self.criterion(self.fuse_out(compressed_z), z)
        output = {
            'z': compressed_z,
            'loss': loss
        }
        return output


class GanFusionSingle(nn.Module):
    """docstring for GanFusionSingle: fusion module for one target modality"""
    def __init__(self,
                 config,
                 generator,
                 discriminator,
                 input_features):
        """
        target_latent: target modality's latent code
        compl_latent: complementary modalities' latent codes (concatenated)
        """
        super(GanFusionSingle, self).__init__()
        self.config = config
        self.auto_fuse = AutoFusion(config,
                                    input_features)
        self.generator = generator
        self.discriminator = discriminator
        self.gen_metrics = None
        self.disc_metrics = None

    def forward(self, z_dict, mode, stage=None):
        # refer to the paper for conventions
        if mode != 'train':
            stage = 'gen'

        if stage == 'disc_real':
            # pass compl_latent through AutoFusion Module
            autofusion_output_dict = self.auto_fuse(self.config,
                                                    z_dict['compl'])
            z_tr = autofusion_output_dict['z']
            auto_fusion_loss = autofusion_output_dict['loss']
            labels = torch.ones_like(z_tr)
            output = self.discriminator(z_tr, labels)
            # add autofusion's loss as well
            output['loss'] += auto_fusion_loss
            output['z_g'] = self.generator(z_dict['target'])['z_g']
            self.disc_metrics = {
                'dral': auto_fusion_loss,
                'drl': output['loss'],
                'dracc': output['accuracy']
            }
        elif stage == 'disc_fake':
            # get the generated latent z_g
            z_g = self.generator(z_dict['target'])['z_g']
            labels = torch.zeros_like(z_g)
            output = self.discriminator(z_g, labels)
            output['z_g'] = z_g
            self.disc_metrics = {
                'dfl': output['loss'],
                'dfacc': output['accuracy']
            }
        elif stage == 'gen':
            output = self.generator(z_dict['target'], self.discriminator)
            self.gen_metrics = {
                'gl': ouptut['loss'],
                'gacc': output['accuracy']
            }
        else:
            raise ValueError(f'invalid stage: {stage}')
        return output


class GanFusion(nn.Module):
    """docstring for GanFusion: contains GAN-Fusion module for all the
    modalities"""
    def __init__(self, config):
        super(GanFusion, self).__init__()
        self.text_gan = GanFusionSingle(config,
                                        GanGenerator(config),
                                        GanDiscriminator(config['latent_dim'],
                                                        config['hidden_dim']))
        self.speech_gan = GanFusionSingle(config,
                                          GanGenerator(config),
                                          GanDiscriminator(config['latent_dim'],
                                                        config['hidden_dim']))
        self.video_gan = GanFusionSingle(config,
                                         GanGenerator(config),
                                         GanDiscriminator(config['latent_dim'],
                                                        config['hidden_dim']))

        self.ff_input_features = config['latent_dim']*3
        self.feed_forward = nn.Sequential(
            nn.Linear(self.ff_input_features, self.ff_input_features//2),
            nn.Tanh(),
            nn.Linear(self.ff_input_features//2, config['latent_dim']),
            nn.ReLU()
            )

    def z_fuse(self, fusion_dict):
        z_fuse_t = fusion_dict['text_dict']['z_g']
        z_fuse_s = fusion_dict['speech_dict']['z_g']
        z_fuse_v = fusion_dict['video_dict']['z_g']
        return self.feed_forward(torch.cat((z_fuse_t,
                                            z_fuse_s,
                                            z_fuse_v), dim=-1))

    def get_loss(self, fusion_dict):
        return fusion_dict['text_dict']['loss'] + \
               fusion_dict['speech_dict']['loss'] + \
               fusion_dict['video_dict']['loss']

    def forward(self, latent_dict, mode, stage=None):
        z_t = latent_dict['z_t']
        z_s = latent_dict['z_s']
        z_v = latent_dict['z_v']

        fusion_dict = {
            'text_dict': self.text_gan({'target': z_t, 'compl': torch.cat((z_s, z_v), dim=-1)}, mode, stage),
            'speech_dict': self.speech_gan({'target': z_s, 'compl': torch.cat((z_t, z_v), dim=-1)}, mode, stage),
            'video_dict': self.video_gan({'target': z_v, 'compl': torch.cat((z_t, z_s), dim=-1)}, mode, stage)
        }

        return {
            'z': self.z_fuse(fusion_dict),
            'loss': self.get_loss(fusion_dict)
        }
