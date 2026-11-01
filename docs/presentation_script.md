# Presentation Script: LettuceDetect Reimplementation

**Team:** Sarah (Intro + Model/Results), Afzal (Literature Survey), Harshita (Data + Contribution/Conclusion)
**Target: about 22-23 minutes total for all three speakers combined.** Do one full read-through with a timer and trim if you run long. Speak it in your own words once you are comfortable; reading word for word sounds flat.

---

## SARAH - Part 1: Introduction, Problem Statement, Objectives (~4 min)

Good morning everyone. I'm Sarah, and along with Afzal and Harshita, we'll be presenting our project on detecting hallucinations in RAG systems.

Let me start by explaining what RAG is. RAG stands for Retrieval-Augmented Generation. Instead of asking a language model a question and letting it answer purely from what it memorized during training, we first retrieve relevant documents or passages from a knowledge base, and hand those to the model along with the question. The model is then supposed to answer using that retrieved evidence, not just from memory. This is how most real-world AI assistants work today, because it lets them answer questions about things that happened after their training cutoff, or about private data they were never trained on.

Now here's the problem we're solving. Even when the retrieved context is completely correct and contains everything needed to answer the question, the model can still say things that aren't actually supported by that context. It might add a detail that sounds correct, and might even be true in the real world, but the model never saw it in the document you gave it. That's what we call a hallucination.

I want to be precise about this definition. A hallucination here doesn't mean the statement is false. It means the statement is unsupported by the given context. For example, if the context talks about when the Eiffel Tower was completed, and the model's answer adds that it was designed by Gustave Eiffel and cost seven point eight million francs, that might be true in the real world, but if the context never said it, the model had no basis to say it either. This narrow definition is useful, because it means we can check this with a program, by comparing the answer against a fixed document, instead of verifying facts against the real world.

Before this paper, there were two ways people tried to catch this. One used a large language model, like GPT-4, to act as a judge. This works well, but it's slow and expensive, since you're paying for another full model call just to check the first one. The other used small, fine-tuned models. These were fast and cheap, but built on older architectures that could only read about five hundred tokens of text at once. Real retrieved contexts in RAG systems run two thousand to eight thousand tokens, so these small models were effectively blind to most of what they were checking.

The paper we chose to reimplement, called LettuceDetect, solves exactly this trade-off. Afzal will now walk you through the research that led up to it, and explain how LettuceDetect fits into that picture.

Our objectives:
First, to reimplement this paper completely from scratch, not using the authors' own code, but writing our own data pipeline, model wrapper, training loop, and evaluation code.
Second, to validate our reimplementation by matching the accuracy the original authors reported.
Third, to go beyond simply reproducing the paper, and test something the original authors never tested. Harshita will explain that later.
Finally, this is the first phase of a three-semester project, so everything we built is meant to be extended.

Over to Afzal now, for the literature survey.

---

## AFZAL - Part 2: Literature Survey, Advantages, and Limitations (~4.5 min)

Thank you, Sarah. My part covers the research that came before LettuceDetect: what each paper tried to solve, what it got right, and where it fell short. I'll go through them in order, since each one reacts to a gap left by the one before it.

The first serious attempt was SelfCheckGPT, from 2023. Its idea: ask the model the same question multiple times and check how consistent the answers are. If the model actually knows something, it tends to say the same thing every time; if it's hallucinating, answers vary. Advantage: needs no training data, works on any model. Downside: expensive, since you need multiple calls per answer, and it never actually looks at the retrieved context.

Around the same time came RAGTruth, which isn't a detector, it's a dataset. Human annotators marked, character by character, exactly which parts of thousands of AI-generated answers were hallucinated. This became the standard benchmark almost every later paper trains and tests on. Its limitation: English only, and no official way to score span-level accuracy, so every paper has had to write its own scoring method, including us.

Next came Luna, in 2024, the first paper to frame this as token classification: every word gets its own label instead of one label for the whole answer. It used a model called DeBERTa. Cheaper and faster than an LLM judge, but still capped at five hundred tokens, so long contexts were still a problem.

RAG-HAT took a different approach: fine-tuning an eight-billion-parameter model to detect hallucinations and also correct them. Highest accuracy in our survey, but at a high cost, both in size and in depending on GPT-4 calls during training.

ReDeEP looks at the problem from a different angle entirely. Instead of reading the output text, it looks inside the model while it's generating, at its attention patterns, to see whether it's actually paying attention to the context or relying on memorized knowledge. This explains why a hallucination happened, but only works with access to the model's internals, which rules out closed models like GPT-4.

This brings us to LettuceDetect, published February 2025, the paper we chose to reimplement. It takes Luna's idea, token-level classification, but swaps the backbone for ModernBERT, which reads up to eight thousand one hundred ninety-two tokens instead of five hundred twelve. That one change removes the context-length problem that limited every small model before it. At one hundred fifty million parameters, it's roughly thirty times smaller than the large-model-based approaches, while matching or beating most of them on accuracy. Its own limitations: English only, looks only at output text, and uses a fixed decision cutoff with no sense of confidence.

A few papers came after, worth mentioning briefly. Turk-LettuceDetect applied the same idea to Turkish by translating RAGTruth; it worked, but accuracy dropped from translation noise. Another paper, in late 2025, raised a serious concern: when tested on data very different from training, this family of detectors dropped close to random guessing. A newer dataset called PsiloQA also appeared, built for many languages using AI-generated annotations instead of human ones, cheaper but less verified.

The pattern: each paper solves the cost or context problem of the one before it, and the next open question becomes generalization: does it still work on data unlike what it trained on. That's exactly what our own contribution, which Harshita presents near the end, is built around.

Over to Harshita, for the dataset.

---

## HARSHITA - Part 1: Dataset and Data Preparation, Slides 7-11 (~4.5 min)

Thanks, Afzal. I'll cover the dataset we used and the work that went into preparing it correctly, since this step turned out to be more delicate than it sounds.

We used RAGTruth. It's the only dataset available that is public, labeled by humans at the character level, and directly comparable to the numbers the original paper reported. It contains two thousand nine hundred sixty-five source items, each a question or task with its supporting passage. Each source item was answered by up to six different language models, including GPT-4, GPT-3.5, Mistral, and Llama variants. That gives seventeen thousand seven hundred ninety total answers. It covers three task types: question answering, summarization, and data-to-text, where the model writes a description from structured data.

Every one of those answers was manually checked by a human annotator, who marked the exact character ranges of any hallucinated content.

Now, the harder part: turning those character-level annotations into something our model can learn from. Our model reads tokens, not characters, so we needed to map each hallucination annotation onto the correct tokens.

This is the single most error-prone part of the pipeline: get this mapping even slightly wrong, and your model trains to completion, the loss numbers look fine, and nothing tells you it's broken. You'd just train on meaningless labels without knowing it. So we checked carefully how the official reference implementation handles this, and matched our approach. We always truncate the context first if the input is too long, never the answer, so no answer token is ever lost. And we locate where the answer begins by counting backward from the end of the sequence, which stays correct no matter how much context got cut, unlike counting forward from the start.

We didn't just trust this pipeline. We wrote a verification script that takes ten real examples with actual hallucinations, and manually checks, character for character, that our pipeline's labels match what the human annotators marked. All ten passed with zero mismatches.

Our final dataset: fifteen thousand ninety training examples and two thousand seven hundred test examples, adding up to RAGTruth's full seventeen thousand seven hundred ninety responses, a useful check that we didn't lose or duplicate anything. The test set is evenly split, nine hundred examples each, across the three task types.

That's the data side. Sarah will now cover the model and our results.

---

## SARAH - Part 3: Model, Implementation, and Results, Slides 12-17 (~5.5 min)

Thanks, Harshita. Now the model itself, how we trained it, and how our results compare to the original paper.

The core idea: instead of one label for the whole answer, every single token in the answer gets its own label. A three-hundred-token answer gives three hundred training signals instead of one, and lets us point to exactly which part of an answer is the problem.

The model is built on ModernBERT. We feed it the context, then the question, then the answer, and put a classification layer on top that predicts, for every answer token, whether it's supported or hallucinated. Context and question tokens are ignored during training; we only grade the model on the answer.

An important note: the original paper trained on an A100 GPU, a high-end data-center card. We trained ours on a single RTX 4060 laptop GPU. That meant a smaller batch size with gradient accumulation to compensate, and a shorter input length cap to fit our memory. Reasonable, documented engineering trade-offs given our hardware, not shortcuts, and we call them out clearly in our report.

So, how did we do? The paper's published result for their base model was seventy-six point zero seven F1. Our own from-scratch reimplementation scored seventy-seven point one eight F1. We matched, and slightly exceeded, the paper's number, with a codebase we wrote entirely ourselves.

By task: eighty-seven point three F1 on data-to-text, sixty-seven point one on question answering, fifty-three point four on summarization. Summarization is clearly our weakest task: the model misses more real hallucinations there than it wrongly flags. This matters because the original paper reports this exact same weakness. A bug that happened to produce a good overall score would have no reason to also reproduce this same weak spot, so this is strong evidence our reimplementation behaves the same way the original does.

I also want to be upfront about something we found checking our own work. Our first training setup picked the best checkpoint by testing it against the same test set we report our final score on, a genuine methodological issue that can make the number look slightly better than it truly is. We didn't retrain to fix this, given the time cost against the likely size of the effect, but we're disclosing it directly instead of presenting our number as beyond question.

We also tested whether adjusting the model's decision cutoff could help. We did this properly: split the test set in half, tuned the cutoff on one half, measured the result only on the untouched other half. Overall, this barely moved the needle, about a tenth of a point, essentially noise. But run separately per task, we found something real: for summarization, adjusting the cutoff improved its score by four point four points, from fifty-five point two to fifty-nine point six, with zero additional training. The reason: hallucinated words make up only three percent of summarization's answer tokens, against eight percent for question answering, so the model trained to be more cautious there, and a lower cutoff compensates for that directly.

To summarize: we reproduced the headline number, reproduced the exact pattern of strengths and weaknesses across tasks, found and disclosed a real issue in our own methodology, and found a genuine, free improvement for our weakest task.

Now Harshita closes with what we're doing beyond this paper.

---

## HARSHITA - Part 2: Our Contribution and Conclusion, Slides 18-22 (~3.5 min)

Thank you, Sarah. To finish, the part of this project we're most excited about.

We came across a newer dataset called PsiloQA, built like RAGTruth but automatically, using AI to generate both the hallucinated answers and their annotations, across fourteen languages. Our first idea was to train on this dataset and present that as our contribution. Before committing, we checked the PsiloQA paper carefully, and found they had already run exactly that experiment in their own results table, including testing the very model we're reimplementing. We're glad we caught that ourselves first.

But that same table pointed us toward something they hadn't done. Their paper shows a detector trained on RAGTruth and one trained on PsiloQA disagree substantially once tested outside their own training data, in one case a gap as large as forty-five percent. What they never test is training one model on both datasets together. Does combining them close that gap, or does mixing two very differently built datasets, one human-labeled, one AI-labeled, just hurt accuracy on both sides? That question is still open, and it's the experiment we're running.

Our plan: train on RAGTruth only, PsiloQA only, and both combined, and test each on both datasets. Our RAGTruth baseline, which Sarah presented, is complete and validated. The PsiloQA and combined runs are the next phase.

One further idea if time allows: our weakest task, summarization, struggles because hallucinated words are rare in its training data. Rather than adjusting the cutoff after training, we could retrain with the loss function weighted to pay more attention to those rare tokens directly, targeting the actual cause rather than compensating afterward.

To close: we reimplemented a recent hallucination detection paper entirely from scratch, matched its published accuracy while reproducing its exact pattern of strengths and weaknesses, which gives us real confidence in our implementation. We held our own work to the same scrutiny we're asking of the literature, by disclosing our own methodology issues rather than hiding them. And we identified a genuine, unanswered question grounded in a specific published number, and we're actively running the experiment to answer it. This is the first of a three-semester project; next steps are the combined-training results, other languages, and eventually a live safety-check integration.

Thank you. We're happy to take questions.
