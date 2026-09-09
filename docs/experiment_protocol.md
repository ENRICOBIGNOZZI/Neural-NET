# Empirical protocol

## Main comparison

The core comparison is deliberately hard to game:

- same architecture at initialization;
- same initialization seed;
- same train/probe/test split;
- same data order or full-batch updates;
- same base optimizer and learning-rate schedule;
- the proposed method may differ only by its contraction decisions;
- test labels are never used by the controller.

## Primary metrics

Report all of the following, not a cherry-picked subset:

1. test loss / accuracy versus **wall-clock**;
2. test loss / accuracy versus **cumulative training FLOPs or a transparent MAC proxy**;
3. active parameter count versus training time;
4. active rank/width per layer versus training time;
5. controller overhead separately;
6. contraction onset time;
7. current target-accessibility loss at every contraction;
8. principal-angle / subspace-motion diagnostics;
9. seed-level results and confidence intervals.

## Controls

At minimum:

- dense AdamW;
- dense AdamW with an optimizer reset at the contraction checkpoint (isolates reset effects);
- static low-rank/width model with the final contracted size from initialization;
- magnitude or structured pruning baseline;
- dynamic sparse baseline (RigL/SRigL where appropriate);
- adaptive-rank baseline for low-rank settings.

## Experiment ladder

### 0. Exact numerical audits
Verify the fixed-geometry theorem to machine precision. No ML benchmark is needed.

### 1. Solvable quadratic teacher--student
The critical falsification test is pruning before versus after dynamic spectral separation. The theory predicts that pre-crossing pruning can delete future signal.

### 2. Small MLP pilot
Digits even/odd and Wisconsin breast cancer. Use 60/20/20 train/probe/test splits. Controller uses train/probe only. First structural action: contract the last hidden layer by neuron subset selection.

### 3. CIFAR / ResNet
Replace neuron contraction with channel contraction. Measure real CUDA wall-clock and FLOPs, not only parameter counts.

### 4. Transformer
Start with fine-tuning, then continued pretraining. Candidate structural units: MLP intermediate width, attention heads, low-rank factors. Compare with GaLore/AdaLoRA-style rank methods only when the setting matches.

## Falsification thresholds

The method fails a benchmark if, after including controller overhead, it does not produce a better performance-vs-compute frontier than the dense baseline within predeclared uncertainty. A lower final parameter count by itself is not sufficient.
