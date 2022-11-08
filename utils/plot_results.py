import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_results(preds, trues):
    pred_len = 1

    plot_index = [i for i in range(0, preds.shape[0], pred_len)]
    plot_preds = preds[plot_index]
    plot_trues = trues[plot_index]

    total_len = plot_preds.shape[0]

    fig1 = plt.figure()
    plt.plot(plot_preds[:].flatten())
    plt.plot(plot_trues[:].flatten())
    plt.legend(['preds', 'trues'])

    fig2 = plt.figure()
    plt.plot(plot_preds[:int(total_len * 0.1)].flatten(), marker='*')
    plt.plot(plot_trues[:int(total_len * 0.1)].flatten(), marker='o')
    plt.legend(['preds', 'trues'])

    fig3 = plt.figure()
    plt.plot(plot_preds[:int(total_len * 0.01)].flatten(), marker='*')
    plt.plot(plot_trues[:int(total_len * 0.01)].flatten(), marker='o')
    plt.legend(['preds', 'trues'])

    fig4 = plt.figure()
    plt.plot(plot_preds[:int(total_len * 0.001)].flatten(), marker='*')
    plt.plot(plot_trues[:int(total_len * 0.001)].flatten(), marker='o')
    plt.legend(['preds', 'trues'])

    fig5 = plt.figure()
    plt.plot(plot_preds[:int(total_len * 0.0001)].flatten(), marker='*')
    plt.plot(plot_trues[:int(total_len * 0.0001)].flatten(), marker='o')
    plt.legend(['preds', 'trues'])

    figs = [fig1, fig2, fig3, fig4, fig5]
    return figs
