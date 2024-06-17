import clrs
import numpy as np
import jax
import jax.numpy as jnp
import pprint as ppr
import datetime
import os

print('Início: ', end='') 
print(datetime.datetime.now())

my_checkpoint_path = './tmp/checkpt/'
best_model_name = 'best.pkl'

# Set random seed and initialize RNG
rng = np.random.RandomState(1234)
rng_key = jax.random.PRNGKey(rng.randint(2**32))

# Algorithm type
algorithm_type = 'quicksort'

# Build samplers
train_sampler, spec = clrs.build_sampler(
    name=algorithm_type,
    num_samples=100,
    length=16
)

test_sampler, spec = clrs.build_sampler(
    name=algorithm_type,
    num_samples=100,
    length=64
)

print("Spec:")
ppr.pprint(spec)

def _iterate_sampler(sampler, batch_size):
    while True:
        yield sampler.next(batch_size)

train_sampler = _iterate_sampler(train_sampler, batch_size=16)
test_sampler = _iterate_sampler(test_sampler, batch_size=32)

# Create processor factory and model parameters
processor_factory = clrs.get_processor_factory('mpnn', use_ln=True)
model_params = dict(
    processor_factory=processor_factory,
    hidden_dim=32,
    encode_hints=True,
    decode_hints=True,
    decode_diffs=False,
    hint_teacher_forcing_noise=1.0,
    use_lstm=False,
    learning_rate=0.0005,
    checkpoint_path=my_checkpoint_path,  
    freeze_processor=False,
    dropout_prob=0.0,
)

# Create model
dummy_trajectory = next(train_sampler)

model = clrs.models.BaselineModel(
    spec=spec,
    dummy_trajectory=dummy_trajectory,
    **model_params
)
model.init(dummy_trajectory.features, 1234)

# Load the last checkpoint if it exists
if os.path.isfile(os.path.join(my_checkpoint_path, best_model_name)):  
    model.restore_model(best_model_name)  
    print('Previous model restored')
else:
    print('Creating new model')

# Training loop
step = 0
best_score = 0
while step <= 100: 
    feedback, test_feedback = next(train_sampler), next(test_sampler)
    rng_key, new_rng_key = jax.random.split(rng_key)
    cur_loss = model.feedback(rng_key, feedback)
    rng_key = new_rng_key
    
    if step % 1 == 0:
        predictions_val, _ = model.predict(rng_key, feedback.features)
        out_val = clrs.evaluate(feedback.outputs, predictions_val)
        predictions, _ = model.predict(rng_key, test_feedback.features)
        out = clrs.evaluate(test_feedback.outputs, predictions)
        print(f'step = {step} | loss = {cur_loss} | val_acc = {out_val["score"]} | test_acc = {out["score"]}')
        print(datetime.datetime.now())
        
        # Store the best model based on validation accuracy
        if (out_val["score"] > best_score):
            best_score = out_val["score"]
            print('Checkpointing best model')
            model.save_model(best_model_name)  
    
    step += 1

print('Fim: ', end='') 
print(datetime.datetime.now())
