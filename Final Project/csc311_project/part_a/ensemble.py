# TODO: complete this file.
from item_response import *


def bagging(data, size):
    resamples = []
    for i in range(size):
        resample = {}

        random_ids = np.random.randint(len(data["user_id"]),
                                       size=len(data["user_id"]))
        resample["user_id"] = np.array(data["user_id"])[random_ids]
        resample["question_id"] = np.array(data["question_id"])[random_ids]
        resample["is_correct"] = np.array(data["is_correct"])[random_ids]
        resamples.append(resample)

    return resamples


def ensemble_evaluate(data, thetas, betas, n_resample):
    # initialize a matrix to store the prediction for all resamples
    preds = np.zeros((n_resample, len(data["question_id"])))
    for j in range(n_resample):
        pred = []
        for i, q in enumerate(data["question_id"]):
            u = data["user_id"][i]
            x = (thetas[j][u] - betas[j][q]).sum()
            p_a = sigmoid(x)
            pred.append(p_a >= 0.5)
        preds[j] = np.array(pred)
    final_pred = np.zeros(len(data["question_id"]))
    mask = np.where(np.sum(preds, axis=0) / n_resample > 0.5)
    final_pred[mask] = 1
    return np.sum((data["is_correct"] == final_pred)) / len(data["is_correct"])


def main():
    train_data = load_train_csv("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    lr = 0.01
    num_iteration = 50

    # need 3 resamples for this question
    n_resamples = 3
    resamples = bagging(train_data, n_resamples)
    thetas = []
    betas = []
    val_accs = []
    test_accs = []

    # firstly, let's run IRT model for each resample data
    for i in range(n_resamples):
        theta, beta, val_acc_lst, train_neg_lld, val_neg_lld = \
            irt(resamples[i], val_data, lr, num_iteration)
        thetas.append(theta)
        betas.append(beta)
        val_acc = evaluate(data=val_data, theta=theta, beta=beta)
        test_acc = evaluate(data=test_data, theta=theta, beta=beta)
        val_accs.append(val_acc)
        test_accs.append(test_acc)


    # now, run predictions with bagging

    val_acc = ensemble_evaluate(val_data, thetas, betas, n_resamples)
    test_acc = ensemble_evaluate(test_data, thetas, betas, n_resamples)
    val_accs.append(val_acc)
    test_accs.append(test_acc)


    # finally test the original training data

    theta, beta, val_acc_lst, train_neg_lld, val_neg_lld = \
        irt(train_data, val_data, lr, num_iteration)
    val_acc = evaluate(data=val_data, theta=theta, beta=beta)
    test_acc = evaluate(data=test_data, theta=theta, beta=beta)
    val_accs.append(val_acc)
    test_accs.append(test_acc)

    print('-----------------------------------------------------------')
    print('Result for IRT without bagging:')
    print('Final validation accuracy: {}'.format(val_accs[0]))
    print('Final test accuracy: {}'.format(test_accs[0]))
    print('----------------------------------')
    print('Final validation accuracy: {}'.format(val_accs[1]))
    print('Final test accuracy: {}'.format(test_accs[1]))
    print('----------------------------------')
    print('Final validation accuracy: {}'.format(val_accs[2]))
    print('Final test accuracy: {}'.format(test_accs[2]))
    print('-----------------------------------------------------------')
    print('Result for IRT with bagging:')
    print('Final ensemble validation accuracy: {}'.format(val_accs[3]))
    print('Final ensemble test accuracy: {}'.format(test_accs[3]))
    print('-----------------------------------------------------------')
    print('Result for IRT original training data :')
    print('Final validation accuracy: {}'.format(val_accs[4]))
    print('Final test accuracy: {}'.format(test_accs[4]))







if __name__ == "__main__":
    main()
