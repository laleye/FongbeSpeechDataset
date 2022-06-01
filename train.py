import os, pandas as pd, numpy as np
from data.dataset import FongbeSpeechDataset
from utils import DataCollatorCTCWithPadding
from arguments import training_args
from datasets import load_dataset, DatasetDict, load_metric
import torchaudio, torch
from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2Processor, Wav2Vec2ForCTC, Trainer




class FongbeASR(object):

    def __init__(self, vocab_file, dataset, device):
        self.tokenizer = Wav2Vec2CTCTokenizer(vocab_file, unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")

        self.feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=False, padding=True)
        self.processor = Wav2Vec2Processor(feature_extractor=self.feature_extractor, tokenizer=self.tokenizer)
        self.dataset = dataset
        self.dataset.split_train_test()
        self.dataset.convert_to_ids(self.processor)
        self.data_collator = DataCollatorCTCWithPadding(processor=self.processor, padding=True)
        self.wer_metric = load_metric("wer")
        self.model = Wav2Vec2ForCTC.from_pretrained(
                    "facebook/wav2vec2-large-xlsr-53",
                    ctc_loss_reduction="mean",
                    attention_dropout=0.1,
                    hidden_dropout=0.1,
                    feat_proj_dropout=0.0,
                    mask_time_prob=0.05,
                    layerdrop=0.1,
                    gradient_checkpointing=True,
                    pad_token_id=self.processor.tokenizer.pad_token_id,
                    vocab_size=len(self.processor.tokenizer))

        print(self.model)
        self.model.freeze_feature_extractor()
        self.model = self.model.to(device)
        self.device = device

    def compute_metrics(self, pred):
        pred_logits = pred.predictions
        pred_ids = np.argmax(pred_logits, axis=-1)

        pred.label_ids[pred.label_ids == -100] = self.processor.tokenizer.pad_token_id

        pred_str = self.processor.batch_decode(pred_ids)
        # we do not want to group tokens when computing the metrics
        label_str = self.processor.batch_decode(pred.label_ids, group_tokens=False)

        wer = self.wer_metric.compute(predictions=pred_str, references=label_str)

        return {"wer": wer}

    def run(self):
        self.trainer = Trainer(
                model=self.model,
                data_collator=self.data_collator,
                args=training_args,
                compute_metrics=self.compute_metrics,
                train_dataset=self.dataset.train_data,
                eval_dataset=self.dataset.eval_data,
                tokenizer=self.processor.feature_extractor,
            )
        self.trainer.train()


if __name__ == "__main__":

    data_dir = "./dataset"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fongbe = FongbeSpeechDataset(data_dir, device)
    fongbe = fongbe.get_dataset()

    vocab = fongbe.get_vocab()

    fasr = FongbeASR("./vocab.json", fongbe, device)

    fasr.run()



