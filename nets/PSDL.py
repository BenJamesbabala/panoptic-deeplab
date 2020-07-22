import torch
import torch.nn as nn

from nets import ResnetBackbone, semContextBlock, insContextBlock, semDecoder, insDecoder, semPrediction, insPrediction, insRegression



class PSDL(nn.Module):
	"""
	docstring for PSDL
	"""
	def __init__(self, num_classes=19): # num_classes = 19 for CityScape data
		super(PSDL, self).__init__()

		self.num_classes = num_classes
		self.backbone = ResnetBackbone()
		self.semcontext = semContextBlock(in_channels=2048)
		self.inscontext = insContextBlock(in_channels=2048)
		self.semdecoder = semDecoder(in_channels_context=256*5, in_channels_block1=256, in_channels_block2=512)
		self.insdecoder = insDecoder(in_channels_context=256*5, in_channels_block1=256, in_channels_block2=512)
		self.semprediction = semPrediction(num_classes=self.num_classes)
		self.insprediction = insPrediction(num_classes=1)
		self.insregression = insRegression(num_classes=2)


	def forward(x):
		
		x1, x2, x_feat = self.backbone(x)
		x_semcon = self.semcontext(x_feat)
		x_inscon = self.inscontext(x_feat)
		x_semdec = self.semdecoder(x_semcon, x2, x1)
		x_insdec = self.insdecoder(x_inscon, x2, x1)
		x_sempred = self.semprediction(x_semdec)
		x_inspred = self.insprediction(x_insdec)
		x_insreg = self.insregression(x_insdec)

		return x_sempred, x_inspred, x_insreg








