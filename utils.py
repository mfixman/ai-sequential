from torch.nn.utils.rnn import pad_sequence
import yaml

def load_config(config_path='config.yaml'):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def collate_fn(batch):
    text_tensors, summary_tensors = zip(*batch)

    text_tensors_padded = pad_sequence(text_tensors, batch_first=True, padding_value=0)
    summary_tensors_padded = pad_sequence(summary_tensors, batch_first=True, padding_value=0)
    
    return text_tensors_padded, summary_tensors_padded
    
def collate_fn_v2(batch):
    text_tensors, summary_tensors, tf_text_tensors, tf_summary_tensors, idf_text_tensors, idf_summary_tensors = zip(*batch)
        
    text_tensors_padded = pad_sequence(text_tensors, batch_first=True, padding_value=0)
    summary_tensors_padded = pad_sequence(summary_tensors, batch_first=True, padding_value=0)
    tf_text_tensors_padded = pad_sequence(tf_text_tensors, batch_first=True, padding_value=0)
    tf_summary_tensors_padded = pad_sequence(tf_summary_tensors, batch_first=True, padding_value=0)
    idf_text_tensors_padded = pad_sequence(idf_text_tensors, batch_first=True, padding_value=0)
    idf_summary_tensors_padded = pad_sequence(idf_summary_tensors, batch_first=True, padding_value=0)

    return text_tensors_padded, summary_tensors_padded, tf_text_tensors_padded, tf_summary_tensors_padded, idf_text_tensors_padded, idf_summary_tensors_padded