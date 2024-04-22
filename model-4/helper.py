import matplotlib.pyplot as pyplot
from IPython import display as dsp

pyplot.ion()

def plot(scores, avg_scores):
    dsp.clear_output(wait=True)
    dsp.display(pyplot.gcf())
    pyplot.clf()
    pyplot.xlabel('No of Games')
    pyplot.ylabel('Score')
    pyplot.title('Training')
    pyplot.plot(scores)
    pyplot.plot(avg_scores)
    pyplot.ylim(ymin=0)
    pyplot.text(len(scores)-1, scores[-1], str(scores[-1]))
    pyplot.text(len(avg_scores)-1, avg_scores[-1], str(avg_scores[-1]))
    pyplot.show(block=False)
    pyplot.pause(.1)