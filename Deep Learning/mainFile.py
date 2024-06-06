import clrs
import numpy as np
import jax
import jax.numpy as jnp
import pprint as ppr
import datetime
import os.path

print('Início: ', end='') 
print(datetime.datetime.now())

# my_checkpoint_path = '/home/felipe/Me/doutorado/quicksort_comparison/tmp/checkpt/'
my_checkpoint_path = './tmp/checkpt/'

best_model_name = 'pgn_10000.pkl'

rng = np.random.RandomState(1234)
rng_key = jax.random.PRNGKey(rng.randint(2**32))

algorithm_type = 'quicksort'
    
train_sampler, spec = clrs.build_sampler(
    name=algorithm_type,
    num_samples=100,
    length=16
)

test_sampler, spec = clrs.build_sampler(
    name=algorithm_type,
    num_samples=100,
    length=32
)

print("Spec:")
ppr.pprint(spec)

def _iterate_sampler(sampler, batch_size):
    while True:
        yield sampler.next(batch_size)

train_sampler = _iterate_sampler(train_sampler, batch_size=16)
test_sampler = _iterate_sampler(test_sampler, batch_size=32)

proccessor_factory = clrs.get_processor_factory('pgn', use_ln=True)
model_params = dict(
    processor_factory=proccessor_factory,
    hidden_dim=64,
    encode_hints=True,
    decode_hints=True,
    decode_diffs=False,
    hint_teacher_forcing_noise=1.0,
    use_lstm=False,
    learning_rate=0.0005,
    checkpoint_path= my_checkpoint_path,
    freeze_processor=False,
    dropout_prob=0.0,
)

dummy_trajectory = next(train_sampler)

model = clrs.models.BaselineModel(
    spec=spec,
    dummy_trajectory=dummy_trajectory,
    **model_params
)
model.init(dummy_trajectory.features, 1234)

#carrega o último checkpoint
if os.path.isfile(my_checkpoint_path + best_model_name):
    model.restore_model(best_model_name)
    print('Previous model restored')
else:
    print('Creating new model')

print('step;loss;val_acc;test_acc;timestamp')
step = 0
best_score = 0
while step <= 10000:
    feedback, test_feedback = next(train_sampler), next(test_sampler)
    rng_key, new_rng_key = jax.random.split(rng_key)
    cur_loss = model.feedback(rng_key, feedback)
    rng_key = new_rng_key
    if step % 1 == 0:
        predictions_val, _ = model.predict(rng_key, feedback.features)
        out_val = clrs.evaluate(feedback.outputs, predictions_val)
        predictions, _ = model.predict(rng_key, test_feedback.features)
        out = clrs.evaluate(test_feedback.outputs, predictions)
        print(f'{step};{cur_loss};{out_val["score"]};{out["score"]};{datetime.datetime.now()}')
    step += 1

    #armazena os pesos da melhor execução
    if (out_val["score"] > best_score) or step == 0:
        best_score = out_val["score"]
        # print('Checkpointing best model')
        model.save_model(best_model_name)

        #ver https://github.com/google-deepmind/clrs/blob/master/clrs/examples/run.py

print('Fim: ', end='') 
print(datetime.datetime.now())
