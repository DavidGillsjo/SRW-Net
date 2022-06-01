import torch
import numpy as np

NOSEM_LABELS = ['invalid', 'valid']

class LabelMapper:
    def __init__(self, line_labels, junction_labels, disable = False):
        if disable:
            self.line_labels = NOSEM_LABELS
            self.junction_labels = NOSEM_LABELS
            self.map = lambda l: (l > 0).astype(l.dtype)
        else:
            self.line_labels = line_labels
            self.junction_labels = junction_labels
            self.map = lambda l: l

    def get_line_labels(self):
        return self.line_labels

    def get_junction_labels(self):
        return self.junction_labels

    def nbr_line_labels(self):
        return len(self.get_line_labels())

    def nbr_junction_labels(self):
        return len(self.get_junction_labels())

    def map_lines(self, labels):
        return self.map(labels)

    def map_junctions(self, labels):
        return self.map(labels)
