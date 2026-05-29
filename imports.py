import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
import torch.nn as nn
import torch.optim as optim
from torch.optim import AdamW
from torcheval.metrics.functional import binary_auprc, binary_auroc
from torchvision.utils import make_grid

import os
from PIL import Image, ImageFile

import numpy as np
import timm
from typing import List, Dict
from sklearn.metrics import classification_report, confusion_matrix
import random
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, train_test_split
import copy

import copy
from typing import List, Dict
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

import cv2
import zipfile
import io

from pytorch_grad_cam import GradCAM, EigenCAM, FinerCAM, XGradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image

import gc

import statistics
import math
from scipy import stats
from docx import Document
from docx.shared import Pt, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_ALIGN_VERTICAL