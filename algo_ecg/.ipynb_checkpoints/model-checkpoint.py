import logging

import keras.layers
import keras.models
import keras.utils

logger = logging.getLogger()


def __format_conv_tuple(tpl, layers):
    '''Takes a tuple or number and a number of conv layers and
    1) Checks that the input is a number or tuple of length either 1 or
       the number of layers
    2) Returns a tuple of length layers which is either the single number
       provided repeated `layers` times, or the tuple provided (when its length
       equals the number of layers)
    '''
    if isinstance(tpl, tuple) and len(tpl) == 1:
        tpl = tuple(tpl[0] for _ in range(layers))
    if isinstance(tpl, int):
        tpl = tuple(tpl for _ in range(layers))
    if len(tpl) != layers:
        raise ValueError('Expected length 1 or {} but got {}'.format(
            layers, len(tpl)
        ))
    return tpl

def create_model(
    dropout=0.,
    optimizer='adam',
    conv_activation='relu',
    conv_filter = (8, 16, 32),
    conv_kernel_size = (12, 12, 12),
    conv_stride = (1, 1, 1),
    conv_pool_size = (1, 1, 4),
    lstm_units = 1
):
    conv_params = {
        'filter': conv_filter,
        'kernel_size': conv_kernel_size,
        'stride': conv_stride,
        'pool_size': conv_pool_size
    }
    conv_layers = 1
    if any(isinstance(x, tuple) for x in conv_params.values()):
        conv_layers = max([
            len(x) for x in conv_params.values() if isinstance(x, tuple)
        ])

    for k, v in conv_params.items():
        conv_params[k] = __format_conv_tuple(conv_params[k], conv_layers)

    model = keras.models.Sequential()
    # model.add(keras.layers.InputLayer(input_shape=(None, 1)))
    model.add(keras.layers.BatchNormalization(
        center=True, scale=False, input_shape=(None, 1)
    ))

    for layer in range(conv_layers):
        model.add(keras.layers.Conv1D(
            conv_params['filter'][layer],
            conv_params['kernel_size'][layer],
            strides=conv_params['stride'][layer],
            padding='same',
            activation=None
        ))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Activation(activation=conv_activation))
        model.add(keras.layers.MaxPool1D(
            pool_size=conv_params['pool_size'][layer]
        ))
        model.add(keras.layers.Dropout(dropout))
    model.add(keras.layers.LSTM(lstm_units, recurrent_dropout=dropout))
    model.add(keras.layers.Dropout(dropout))
    model.add(keras.layers.Dense(1, activation='sigmoid'))
    # model.add(keras.layers.Dense(8, activation='softmax'))
    model.compile(
        loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy']
    )
    # model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
