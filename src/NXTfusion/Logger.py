# Code referenced from https://gist.github.com/gyglim/1f8dfb1b5c82627ae3efcfbbadb9f514
import numpy as np
import scipy.misc
import signal, os, sys
 
def signal_term_handler(signal, frame):
	os.system("killall tensorboard")
	print ('got SIGTERM, killing tensorboard.')
	sys.exit(0)
 
try:
    from StringIO import StringIO  # Python 2.7
except ImportError:
    from io import BytesIO         # Python 3.x

def to_np(x):
    return x.data.cpu().numpy()

def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)   

class MetaLogger(object):

	""" This class incapsulates the tensorboard logging code,
	but uses the underlying Logger class to do all the real work."""

	def __init__(self, model, port = 6006):
		self.model = model
		import tensorflow as tf
		os.system("rm -rf ./logs")
		os.system("killall tensorboard")
		os.system("tensorboard --logdir='./logs' --port="+str(port)+"&")
		self.logger = Logger('./logs')			
		os.system("firefox http://127.0.0.1:"+str(port)+"  &")
		signal.signal(signal.SIGTERM, signal_term_handler)

	def writeTensorboardLog(self, step, errTot, lossScores, relationList, te):
		#step == epoch
		if self.logger == None:
			print (" WARNING: Logger not initializated, skipping")
			return
		info = {"errTot":errTot}
		for ri, r in enumerate(relationList):
			for i, l in enumerate(lossScores[ri]):
				info[r[i]["name"]] = l
		info["timePerEpoch"] = te

		for tag, value in info.items():
			self.logger.scalar_summary(tag, value, step+1)

		# (2) Log values and gradients of the parameters (histogram)
		for tag, value in self.model.named_parameters():
			#print
			#print tag#, value, value.grad					
			tag = tag.replace('.', '/')
			self.logger.histo_summary(tag, to_np(value), step+1)
			self.logger.histo_summary(tag+'/grad', to_np(value.grad), step+1)

		return info

	def shutdown(self):
		print ("Logger: killing tensorboard.")
		os.system("killall tensorboard")


class Logger(object):
    
	"""This class actually writes data on the tensorboard logger. It could be used
	directly or it can be used through the MetaLogger class"""

	def __init__(self, log_dir):
		"""Create a summary writer logging to log_dir."""
		self.writer = tf.summary.FileWriter(log_dir)
		tf.logging.set_verbosity(tf.logging.WARN)
		os.environ['TF_CPP_MIN_LOG_LEVEL']='5'

	def list_summary(self, tags, values, step):
		assert len(tags) == len(values)
		for i, t in enumerate(tags):
			self.scalar_summary(t, values[i], step)

	def scalar_summary(self, tag, value, step):
		"""Log a scalar variable."""
		summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
		self.writer.add_summary(summary, step)

	def image_summary(self, tag, images, step):
		"""Log a list of images."""

		img_summaries = []
		for i, img in enumerate(images):
			# Write the image to a string
			try:
				s = StringIO()
			except:
				s = BytesIO()
			scipy.misc.toimage(img).save(s, format="png")

			# Create an Image object
			img_sum = tf.Summary.Image(encoded_image_string=s.getvalue(),
									   height=img.shape[0],
									   width=img.shape[1])
			# Create a Summary value
			img_summaries.append(tf.Summary.Value(tag='%s/%d' % (tag, i), image=img_sum))

		# Create and write Summary
		summary = tf.Summary(value=img_summaries)
		self.writer.add_summary(summary, step)
		
	def histo_summary(self, tag, values, step, bins=1000):
		"""Log a histogram of the tensor of values."""

		# Create a histogram using numpy
		counts, bin_edges = np.histogram(values, bins=bins)

		# Fill the fields of the histogram proto
		hist = tf.HistogramProto()
		hist.min = float(np.min(values))
		hist.max = float(np.max(values))
		hist.num = int(np.prod(values.shape))
		hist.sum = float(np.sum(values))
		hist.sum_squares = float(np.sum(values**2))

		# Drop the start of the first bin
		bin_edges = bin_edges[1:]

		# Add bin edges and counts
		for edge in bin_edges:
			hist.bucket_limit.append(edge)
		for c in counts:
			hist.bucket.append(c)

		# Create and write Summary
		summary = tf.Summary(value=[tf.Summary.Value(tag=tag, histo=hist)])
		self.writer.add_summary(summary, step)
		self.writer.flush()
