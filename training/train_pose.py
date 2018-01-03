import sys
sys.path.append("..")

from training.train_common import prepare, train, validate, validate_batch, save_network_input_output
from training.ds_generators import DataGeneratorClient, DataIterator
from config import COCOSourceConfig, GetConfig

use_client_gen = False
batch_size = 10

task = sys.argv[1] if len(sys.argv)>1 else "train"
config_name = sys.argv[2] if len(sys.argv)>2 else "Canonical"
experiment_name = sys.argv[3] if len(sys.argv)>3 else None

model, iterations_per_epoch, validation_steps, last_epoch, metrics_id, callbacks_list = \
    prepare(config_name=config_name, exp_id=experiment_name, train_samples = 117576, val_samples = 2475, batch_size=batch_size)

config = GetConfig(config_name)

if use_client_gen:
    train_client = DataGeneratorClient(config, port=5555, host="localhost", hwm=160, batch_size=batch_size)
    val_client = DataGeneratorClient(config, port=5556, host="localhost", hwm=160, batch_size=batch_size)
else:
    train_client = DataIterator(config, COCOSourceConfig("../dataset/coco_train_dataset.h5"), shuffle=True,
                                augment=True, batch_size=batch_size)
    val_client = DataIterator(config, COCOSourceConfig("../dataset/coco_val_dataset.h5"), shuffle=False, augment=False,
                              batch_size=batch_size)

train_di = train_client.gen()
val_di = val_client.gen()


if task == "train":
    train(model, train_di, val_di, iterations_per_epoch, validation_steps, last_epoch, use_client_gen, callbacks_list)

elif task == "validate":
    validate(model, val_di, validation_steps, use_client_gen)

elif task == "validate_batch":
    validate_batch(model, val_di, validation_steps, metrics_id)

elif task == "save_network_input_output":
    save_network_input_output(model, val_di, validation_steps, metrics_id, batch_size)

elif task == "save_network_input":
    save_network_input_output(None, val_di, validation_steps, metrics_id, batch_size)
