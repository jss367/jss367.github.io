---
layout: post
title: "Argparse Notes"
description: "This post shows how to spoof argparse to get modules to work in Jupyter Notebooks"
feature-img: "assets/img/rainbow.jpg"
thumbnail: "assets/img/kea.jpg"
tags: [Jupyter Notebooks, Python]
---

## Required arguments

Arguments that don't begin with "-" or "--" are positional and therefore required (much like arguments in Python functions). Arguments that start with  "-" or "--" are optional arguments, although `argparse` lets you mark this as required. This is considered bad design in most casee but isn't prevented.

For example, you can do:

```python
parser = argparse.ArgumentParser()
parser.add_argument('--foo', required=True)
```


This is how it's put in the documentation [documentation](https://docs.python.org/3/library/argparse.html#required):
> Required options are generally considered bad form because users expect options to be optional, and thus they should be avoided when possible.


## Spoofing Argpase

It's often a good idea to construct deep learning code so that experiments can be conducted simply from the command line, allowing you to quickly iterate and record results. [argparse](https://docs.python.org/3/library/argparse.html) makes this super easy by allowing configurations to be passed as command-line arguments. But sometimes you want to do the opposite. You find a repo that has a `train.py` file ready-to-go but you want to walk through it line-by-line in a Jupyter Notebook.

Unfortunately, `argparse` can get in the way here because the code is expecting an `argparse` object and you're not calling it from the command line. To get around this, you'll have to spoof `argparse`. 

#### Simple Example

If you're going to run a program in, say, VSCode, it's already going to have the file in `sys.argv`, even if you don't add any other arguments. If you want to quickly spoof argparse, you can extend `sys.argv` like so:

`sys.argv.extend(["my_arg_1", "my_arg_2"])`


#### Full Example

For a fuller example, I'll show how to do that using [`centermask2`](https://github.com/youngwanLEE/centermask2) as an example. `centermask2` is an implementation of [CenterMask: Real-Time Anchor-Free Instance Segmentation](https://arxiv.org/abs/1911.06667) based on Facebook AI Research's object detection [detectron2](https://github.com/facebookresearch/detectron2) framework.

Let's start by looking at the [centermask2 code](https://github.com/youngwanLEE/centermask2). If you go to the repository, you'll see that there's a `train_net.py` file. Opening it up, [at the bottom we see the following](https://github.com/youngwanLEE/centermask2/blob/588b2bde8a1a48756a3089190109cdc1f03cdc68/train_net.py#L221):


```python
if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
```

OK, so like other detectron2 modules, what we need to do is call `launch`. But we need to do it with an `args.num_gpus`, `args.num_machines`, `args.machine_rank`, `args.dist_url`, and whatever else is hidden inside `args`. How do we do this?

 Fortunately, the solution is quite simple. **All you need to do is pass your arguments in a list to `sys.argv` and then pass that as an argument to parser.parse_args(sys.argv).** Here's an example:


```python
import sys
import argparse

sys.argv = ["--config-file", "/path/to/configs/Base-CenterMask-VoVNet-Modified.yaml"]

```

Now we can copy them parse that using [detectron2's default_argument_parse](https://github.com/facebookresearch/detectron2/blob/7557b76543f2b1f115b96dc4a9432e5b69140571/detectron2/engine/defaults.py#L49). I've copied it below for reference.


```python
def default_argument_parser(epilog=None):
    """
    Create a parser with some common arguments used by detectron2 users.
    Args:
        epilog (str): epilog passed to ArgumentParser describing the usage.
    Returns:
        argparse.ArgumentParser:
    """
    parser = argparse.ArgumentParser(
        epilog=epilog
        or f"""
Examples:
Run on single machine:
    $ {sys.argv[0]} --num-gpus 8 --config-file cfg.yaml MODEL.WEIGHTS /path/to/weight.pth
Run on multiple machines:
    (machine0)$ {sys.argv[0]} --machine-rank 0 --num-machines 2 --dist-url <URL> [--other-flags]
    (machine1)$ {sys.argv[0]} --machine-rank 1 --num-machines 2 --dist-url <URL> [--other-flags]
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config-file", default="", metavar="FILE", help="path to config file")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="whether to attempt to resume from the checkpoint directory",
    )
    parser.add_argument("--eval-only", action="store_true", help="perform evaluation only")
    parser.add_argument("--num-gpus", type=int, default=1, help="number of gpus *per machine*")
    parser.add_argument("--num-machines", type=int, default=1, help="total number of machines")
    parser.add_argument(
        "--machine-rank", type=int, default=0, help="the rank of this machine (unique per machine)"
    )

    # PyTorch still may leave orphan processes in multi-gpu training.
    # Therefore we use a deterministic way to obtain port,
    # so that users are aware of orphan processes by seeing the port occupied.
    port = 2 ** 15 + 2 ** 14 + hash(os.getuid() if sys.platform != "win32" else 1) % 2 ** 14
    parser.add_argument(
        "--dist-url",
        default="tcp://127.0.0.1:{}".format(port),
        help="initialization URL for pytorch distributed backend. See "
        "https://pytorch.org/docs/stable/distributed.html for details.",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser
```

Before you would pass your arguments by calling `default_argument_parser().parse_args()` where `parse_args` is given no argument. Now you want to pass `sys.argv` to it as an argument.


```python
args = default_argument_parser().parse_args(sys.argv)
```


```python
print("Command Line Args:", args)
```

    Command Line Args: Namespace(config_file='/orbital/base/core/projects/oidetectron2/configs/Base-CenterMask-VoVNet-Modified.yaml', dist_url='tcp://127.0.0.1:49153', eval_only=False, machine_rank=0, num_gpus=1, num_machines=1, opts=[], resume=False)
    

That's all there is!


