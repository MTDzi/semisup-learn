from sklearn.metrics import average_precision_score, precision_recall_curve
import matplotlib as mpl
import matplotlib.pyplot as plt


mpl.style.use('seaborn-dark-palette')
mpl.style.use('seaborn-whitegrid')

mpl.rcParams['legend.framealpha'] = 1
mpl.rcParams['legend.frameon'] = True
mpl.rcParams['legend.fontsize'] = 14

mpl.rcParams['figure.figsize'] = (10, 8)
mpl.rcParams['font.size'] = 14

mpl.rcParams['xtick.labelsize'] = 14
mpl.rcParams['ytick.labelsize'] = 14

mpl.rcParams['axes.labelsize'] = 18
mpl.rcParams['axes.titlesize'] = 20


def plot_prec_rec_curve(y_test, y_pred, title):
  prec_recall_curve = precision_recall_curve(y_test, y_pred)

  plt.plot(prec_recall_curve[1], prec_recall_curve[0],
           label='{} (PrecRecAUC = {:.2f})'.format(title, average_precision_score(y_test, y_pred.flatten())),
           lw=3)
  plt.fill_between(prec_recall_curve[1], 0, prec_recall_curve[0], alpha=0.3)
  plt.xlabel('Recall')
  plt.ylabel('Precision')
  plt.xlim([0, 1])
  plt.ylim([0, 1])
  plt.legend(loc=1);
