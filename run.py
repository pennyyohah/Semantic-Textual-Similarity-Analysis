from args import Config
from transformers import BertForSequenceClassification,BertConfig
import time
from utils import get_sent,get_data_loader,format_time
from train_eval import train

data_path = './data'
bert_path = './bert'

if __name__ =='__main__':
    config = Config(data_path, bert_path)
    start_time = time.time()

    print("Loading data...")
    train_sent, dev_sent, test_sent = get_sent(data_path)

    train_dataloader = get_data_loader(config, train_sent)
    dev_dataloader = get_data_loader(config, dev_sent)
    test_dataloader = get_data_loader(config, test_sent)
    print("Time usage:", format_time(time.time() - start_time))

    # train
    # configBert = BertConfig.from_json_file(config.bert_path + '/config.json')
    # configBert.num_labels = 2
    model = BertForSequenceClassification.from_pretrained(config.bert_path)

    # model = BertForSequenceClassification.from_pretrained("bert-base-chinese")

    train(config, model, train_dataloader, dev_dataloader, test_dataloader)