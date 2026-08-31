# Model Card - Intent Router on REAL messages

## What this is

The **same pipeline** the service serves -- TF-IDF over word 1-2 grams and
character 3-5 grams, into a logistic regression, `C` selected by 5-fold
cross-validation on the training split only -- fitted on real customer messages
instead of generated ones.

**Data:** REAL -- BANKING77 (CC BY 4.0)
Casanueva, Temcinas, Gerz, Henderson and Vulic, 'Efficient Intent Detection with Dual Sentence Encoders', NLP4ConvAI 2020
https://huggingface.co/datasets/PolyAI/banking77

10003 training and 3080 test messages that real
customers sent an online bank, labelled with 77 fine-grained
intents. The **official split ships with the dataset and is used as shipped**, so
these numbers sit on the same split as published work rather than one we chose.

## Measured performance

| Metric | Value |
|---|---|
| **Macro-F1** | **0.9164** |
| Accuracy | 0.9162 |
| Cross-validated macro-F1 (train) | 0.8981 |

### Why this number was checked before it was published

0.9164 on a 77-class problem is high
enough to be suspicious, so it was tested rather than trusted. Verbatim overlap
between the two official files is
**6 messages**
(0.195% of the test set) -- far too
small to explain the result, and a property of the dataset rather than of any
split we chose.

The test set is balanced at 40 examples per
class, which is why macro-F1 and accuracy nearly coincide here. On an imbalanced
set they would diverge and macro-F1 would be the one to read.

The honest reading is that these queries are lexically distinctive -- customers
asking about a lost card and about a declined transfer use different words -- and
character n-grams handle the typos. That suits a sparse linear model better than
expected.

### Read against the right scale

77 classes, so random guessing scores
0.013 accuracy and always predicting the
most common intent (`card_payment_fee_charged`) scores
0.013.

This is a substantially harder task than the served router's, which separates 5
intents. The two macro-F1 figures are **not comparable** and the READMEs do not
compare them: one is 5-way on our phrasing, the other is 77-way on real phrasing.
What carries across is that the same pipeline handles both.

### Where it fails

Hardest intents by F1:

| Intent | F1 |
|---|---|
| `balance_not_updated_after_bank_transfer` | 0.747 |
| `top_up_failed` | 0.8101 |
| `failed_transfer` | 0.8182 |
| `unable_to_verify_identity` | 0.825 |
| `card_not_working` | 0.8352 |

Easiest:

| Intent | F1 |
|---|---|
| `verify_top_up` | 1.0 |
| `passcode_forgotten` | 1.0 |
| `apple_pay_or_google_pay` | 1.0 |
| `edit_personal_details` | 0.9877 |
| `age_limit` | 0.9877 |

The hard cases are near neighbours rather than random errors -- pairs that differ
by intent rather than by vocabulary, where a bag-of-n-grams model has little to
work with. A sentence encoder would close most of that gap, and the fact that this
model cannot is a property of the representation, not a bug.

## How this relates to the served model

The served router is **not** this model: it classifies 5 intents in a support
domain against this one's 77 in a banking domain. This track validates the
*method* -- that the pipeline learns real customer phrasing, not just phrasing we
wrote -- against messages nobody composed for a classifier.

## Known limitations

- Intent only. BANKING77 has no sentiment labels, so the sentiment component is
  still validated on synthetic data alone.
- One domain (retail banking) and one channel (text chat).
- No calibration analysis; the router's confidence is logistic regression's raw
  output.
- Linear model over sparse n-grams: no word order beyond bigrams, no semantics.
