# Define model

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class Encoder(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.enc_image_size = params.encoded_image_size

        resnet = torchvision.models.resnet101(
            weight=torchvision.models.ResNet101_Weights.IMAGENET1K_V2
        )

        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)

        self.adaptive_pool = nn.AdaptiveAvgPool2d(
            (self.enc_image_size, self.enc_image_size)
        )

        self._fine_tune()

    def forward(self, images):
        # (B, 2048, H/32, W/32)
        out = self.resnet(images)
        # (B, 2048, A, A)
        out = self.adaptive_pool(out)
        # (B, 2048, A, A) -> (B, A, A, 2048)
        out = out.permute(0, 2, 3, 1)

        return out

    def _fine_tune(self, fine_tune=True):
        for p in self.resnet.parameters():
            p.requires_grad = False

        for c in list(self.resnet.children())[5:]:
            for p in c.parameters():
                p.requires_grad = fine_tune


class Attention(nn.Module):

    def __init__(self, params):
        super().__init__()
        self.encoder_att = nn.Linear(params.encoder_dim, params.attention_dim)
        self.decoder_att = nn.Linear(params.decoder_dim, params.attention_dim)
        self.full_att = nn.Linear(params.attention_dim, 1)

    def forward(self, encoder_out, decoder_hidden):
        # (B,N_pixels, E_dim) -> (B, N_pixels, A_dim)
        att1 = self.encoder_att(encoder_out)
        # (B, D_dim) -> (B, A_dim)
        att2 = self.decoder_att(decoder_hidden)
        # (B, N_pixels)
        att = self.full_att(F.relu(att1 + att2.unsqueeze(1))).squeeze(2)
        alpha = F.softmax(att, dim=-1)

        # (B, E_dim)
        attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)

        return attention_weighted_encoding, alpha


class DecoderWithAttention(nn.Module):
    def __init__(self, params):
        super().__init__()

        self.encoder_dim = params.encoder_dim
        self.attention_dim = params.attention_dim
        self.embed_dim = params.embed_dim
        self.decoder_dim = params.decoder_dim
        self.vocab_size = params.vocab_size
        self.p_dropout = params.p_dropout

        self.attention = Attention(
            self.encoder_dim, self.decoder_dim, self.attention_dim
        )

        self.embedding = nn.Embedding(self.vocab_size, self.embed_dim)
        self.dropout = nn.Dropout(p=self.p_dropout)
        self.decode_step = nn.LSTMCell(
            self.embed_dim + self.encoder_dim, self.decoder_dim, bias=True
        )
        self.init_h = nn.Linear(self.encoder_dim, self.decoder_dim)
        self.init_c = nn.Linear(self.encoder_dim, self.decoder_dim)
        self.f_beta = nn.Linear(self.decoder_dim, self.encoder_dim)

        self.fc = nn.Linear(self.decoder_dim, self.vocab_size)

        self._init_weights()

    def _init_weights(self):
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def load_pretrained_embeddings(self, embeddings):
        self.embedding.weight = nn.Parameter(embeddings)

    def fine_tune_embeddings(self, fine_tune=True):
        for p in self.embedding.parameters():
            p.requires_grad = fine_tune

    def init_hidden_state(self, encoder_out):
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out)
        c = self.init_c(mean_encoder_out)
        return h, c

    def forward(self, encoder_out, encoded_captions, caption_lengths, params):
        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)
        vocab_size = self.vocab_size

        # Flatten image
        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)
        num_pixels = encoder_out.size(1)

        # sort input data by decreasing lengths
        caption_lengths, sort_ind = caption_lengths.squeeze(1).sort(
            dim=0, descending=True
        )
        encoder_out = encoder_out[sort_ind]
        encoded_captions = encoded_captions[sort_ind]
        encoded_captions = encoded_captions[sort_ind]

        # Embeddings
        embeddings = self.embedding(encoded_captions)

        # Initialize LSTM state
        h, c = self.init_hidden_state(encoder_out)

        decode_lengths = (caption_lengths - 1).tolist()

        if params.cuda:
            predictions = torch.zeros(batch_size, max(decode_lengths), vocab_size).cuda(
                non_blocking=True
            )
            alphas = torch.zeros(batch_size, max(decode_lengths), num_pixels).cuda(
                non_blocking=True
            )

        # at each time-step, decode
        for t in range(max(decode_lengths)):
            batch_size_t = sum([l > t] for l in decode_lengths)
            attention_weighted_encoding, alpha = self.attention(
                encoder_out[:batch_size_t], h[:batch_size_t]
            )
            gate = F.sigmoid(self.f_beta(h[:batch_size_t]))
            attention_weighted_encoding = gate * attention_weighted_encoding
            h, c = self.decode_step(
                torch.cat(
                    [embeddings[:batch_size_t, t, :], attention_weighted_encoding],
                    dim=1,
                ),
                (h[:batch_size_t], c[:batch_size_t]),
            )
            preds = self.fc(self.dropout(h))
            predictions[:batch_size_t, t, :] = preds
            alphas[:batch_size_t, t, :] = alpha

        return predictions, encoded_captions, decode_lengths, alphas, sort_ind
