# Modified from QuaRot (https://github.com/spcl/QuaRot), Apache License 2.0.
import logging

from accelerate import Accelerator, PartialState
from accelerate.utils import gather_object, wait_for_everyone
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

import quant_utils
from distribute_utils import (
    split_between_selected,
    group_gather_object,
    get_process_group_ranks,
    wait_every_group_member,
)


@torch.inference_mode()
def ppl_eval(model, testloader, batch_size, group_main_ranks=None):
    model.eval()
    device = model.device

    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    state = PartialState()
    nlls = []
    with split_between_selected(testloader, group_main_ranks) as split_testloader:
        dataloader = DataLoader(TensorDataset(split_testloader), batch_size)  # type: ignore
        for batch in (pbar := tqdm(
            dataloader,
            desc=f"[rank {state.process_index}] Evaluating PPL",
        )):
            input_ids = batch[0].to(device)
            lm_logits = model(input_ids).logits
            shift_logits = lm_logits[:, :-1, :].contiguous()
            shift_labels = input_ids[:, 1:]
            loss = loss_fct(shift_logits.permute(0, 2, 1), shift_labels)
            neg_log_likelihood = loss.float().mean(dim=1)
            nlls.extend(neg_log_likelihood.tolist())
            pbar.set_postfix({"ppl": torch.tensor(nlls).mean().exp().item()})
    wait_for_everyone()
    nlls = gather_object(nlls)
    ppl = torch.exp(torch.tensor(nlls).sum() / len(testloader)).item()
    return ppl, nlls
