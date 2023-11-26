def build_bert_model(max_len):
    """ add binary classification to pretrained model
    """

    input_word_ids = tf.keras.layers.Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")

    bert_model = TFBertModel.from_pretrained("bert-base-german-cased", name='pre-trained_bert')
    encoder_outputs = bert_model(input_word_ids)

    last_hidden_state = encoder_outputs[0]
    pooler_output = encoder_outputs[1]
    cls_embedding = pooler_output

                                                                       #x = tf.keras.layers.Dense(18, name = "fc1", activation='relu')(cls_embedding)
    dropout = tf.keras.layers.Dropout(0.1)(cls_embedding)              # improves robustness
    stack = tf.keras.layers.Dense(9, name = "fc")(dropout)             # 9 output neurons (equal to num_distinct_classes)

    output = tf.keras.layers.Activation('softmax')(stack)
    model = tf.keras.models.Model(inputs=input_word_ids, outputs=output)
    return model

def lrfn(epoch, LR_START, LR_MAX, LR_MIN, LR_RAMPUP_EPOCHS, LR_SUSTAIN_EPOCHS, LR_EXP_DECAY):
        if epoch < LR_RAMPUP_EPOCHS:
            lr = (
                LR_MAX - LR_START
            ) / LR_RAMPUP_EPOCHS * epoch + LR_START
        elif epoch < LR_RAMPUP_EPOCHS + LR_SUSTAIN_EPOCHS:
            lr = LR_MAX
        else:
            lr = (LR_MAX - LR_MIN) * LR_EXP_DECAY ** (
                epoch - LR_RAMPUP_EPOCHS - LR_SUSTAIN_EPOCHS
            ) + LR_MIN
        return lr


