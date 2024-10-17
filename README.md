# Atari-GPT

This is the official codebase for [Atari-GPT](https://arxiv.org/pdf/2408.15950), a new benchmark for Large Language Models (LLMs) on Atari games. To see more results see our project [webpage](https://sites.google.com/view/atari-gpt/).

## Set up

In order to run the code you need to have an API key for the respective model. To reproduce results from the paper you will need all 3 API keys.

&ensp;&ensp;&ensp;&ensp;For Google you can get an API key [here](https://ai.google.dev). Once you have this API key put it in a file called GOOGLE_API_KEY.txt.


&ensp;&ensp;&ensp;&ensp;For Anthropic you can get an API key [here](https://www.anthropic.com/api). Once you have this API key put it in a file called ANTHROPIC_API_KEY.txt.

&ensp;&ensp;&ensp;&ensp;For OpenAI you can get an API key [here](https://openai.com/api/). Once you have this API key put it in a file called OPENAI_API_KEY.txt.


## Installation

To run the code you will need to have Anaconda and run the following commands:

<br>&ensp;&ensp;&ensp;&ensp;`conda env create --file=environment.yaml`
<br>&ensp;&ensp;&ensp;&ensp;`conda activate atari_gpt`
<br>&ensp;&ensp;&ensp;&ensp;`python full_evaluation.py`

## Citing Atari-GPT 

```
@misc{waytowich2024atarigptinvestigatingcapabilitiesmultimodal,
      title={Atari-GPT: Investigating the Capabilities of Multimodal Large Language Models as Low-Level Policies for Atari Games}, 
      author={Nicholas R. Waytowich and Devin White and MD Sunbeam and Vinicius G. Goecks},
      year={2024},
      eprint={2408.15950},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2408.15950}, 
}
```