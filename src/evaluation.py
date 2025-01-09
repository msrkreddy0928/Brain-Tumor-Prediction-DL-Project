


def CNN_evaluate(model,test):
    loss, accuracy = model.evaluate(test)
    return loss,accuracy