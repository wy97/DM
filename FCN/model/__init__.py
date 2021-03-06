#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import copy

import torchvision.models as models

from FCN.model.fcn import fcn8s, fcn16s, fcn32s


def get_model(model_dict, n_classes, version=None):

    name = model_dict["arch"]

    model = _get_model_instance(name)

    param_dict = copy.deepcopy(model_dict)

    param_dict.pop("arch")



    if name in ["fcn32s", "fcn16s", "fcn8s"]:

        model = model(n_classes=n_classes, **param_dict)

        vgg16 = models.vgg16(pretrained=True)

        model.init_vgg16_params(vgg16)
    
    return model


def _get_model_instance(name):

    try:

        return {

            "fcn32s": fcn32s,

            "fcn8s": fcn8s,

            "fcn16s": fcn16s

            #"unet": unet,

            #"segnet": segnet,

            #"pspnet": pspnet,

            #"icnet": icnet,

            #"icnetBN": icnet,

            #"linknet": linknet,

            #"frrnA": frrn,

            #"frrnB": frrn,

        }[name]

    except:

        raise ("Model {} not available".format(name))




