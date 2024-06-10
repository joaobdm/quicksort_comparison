import clrs
import numpy as np
import jax
import os.path
import pprint as ppr
import datetime
import time
from sys_monitoring import start_monitoring, stop_monitoring


checkpoint_path = './tmp/checkpt/'
model_name = 'mpnn_10000.pkl'
TEST_ARRAY_LENGTH = 32
NUM_OF_TESTS = 100000

rng = np.random.RandomState(1234)
rng_key = jax.random.PRNGKey(rng.randint(2**32))
# print('rng_key',rng_key)

algorithm_type = 'quicksort'

# Construir sampler para testar o modelo com novos dados
test_sampler, spec = clrs.build_sampler(
    name=algorithm_type,
    num_samples=100,  # Número de amostras de teste
    length=TEST_ARRAY_LENGTH         # Tamanho das amostras de teste
)

# print("Spec:")
# ppr.pprint(spec)

def _iterate_sampler(sampler, batch_size):
    while True:
        yield sampler.next(batch_size)

test_sampler = _iterate_sampler(test_sampler, batch_size=32)

processor_factory = clrs.get_processor_factory('mpnn', use_ln=True)
model_params = dict(
    processor_factory=processor_factory,
    hidden_dim=64,
    encode_hints=True,
    decode_hints=True,
    decode_diffs=False,
    hint_teacher_forcing_noise=1.0,
    use_lstm=False,
    learning_rate=0.0005,
    checkpoint_path=checkpoint_path,
    freeze_processor=False,
    dropout_prob=0.0,
)

dummy_trajectory = next(test_sampler)

# Inicializar o modelo
model = clrs.models.BaselineModel(
    spec=spec,
    dummy_trajectory=dummy_trajectory,
    **model_params
)
model.init(dummy_trajectory.features, 1234)

# Carregar o modelo salvo
if os.path.isfile(checkpoint_path + model_name):
    model.restore_model(model_name)
    # print('Modelo carregado com sucesso')
# else:
#     print('Erro: Modelo salvo não encontrado')

# Testar o modelo com novos dados
def test_model(model, sampler, num_tests):
    accuracies = []
    for _ in range(num_tests):
        feedback = next(sampler)
        predictions, _ = model.predict(rng_key, feedback.features)
        accuracy = clrs.evaluate(feedback.outputs, predictions)
        # print('feedback',feedback)
        # print('predictions',predictions)
        # print('accuracy',accuracy)
        accuracies.append(accuracy["score"])
    return accuracies

# Executar o teste
print('Início: ', end='') 
start = datetime.datetime.now()

print(start)
print(f'MODEL FILE:{model_name}, ALGORITHM TYPE: {algorithm_type}, TEST ARRAY LENGTH: {TEST_ARRAY_LENGTH}, NUMBER OF TESTS: {NUM_OF_TESTS}')
start_log_time = time.time()
start_monitoring(model_name,TEST_ARRAY_LENGTH,NUM_OF_TESTS)
accuracies = test_model(model, test_sampler, num_tests=NUM_OF_TESTS)
end_log_time = time.time()
stop_monitoring(end_log_time - start_log_time)
print("Acurácias das execuções de teste:")
print(accuracies)
print("Acurácia média:", np.mean(accuracies))
print('Fim: ', end='') 
end = datetime.datetime.now()
print(datetime.datetime.now())
print('Tempo decorrido')
print(end - start)