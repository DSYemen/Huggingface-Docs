# الاستدلال باستخدام وحدة المعالجة المركزية

مع بعض التحسينات، يمكن تشغيل الاستدلال على النماذج الكبيرة بكفاءة على وحدة المعالجة المركزية. تنطوي إحدى تقنيات التحسين على تجميع تعليمات PyTorch البرمجية إلى تنسيق وسيط لبيئات عالية الأداء مثل C++. وتقوم التقنية الأخرى بدمج عمليات متعددة في نواة واحدة لتقليل النفقات العامة لتشغيل كل عملية على حدة.

ستتعلم كيفية استخدام BetterTransformer للاستدلال بشكل أسرع، وكيفية تحويل تعليمات PyTorch البرمجية الخاصة بك إلى TorchScript. إذا كنت تستخدم وحدة المعالجة المركزية من Intel، فيمكنك أيضًا استخدام التحسينات الرسومية من Intel Extension for PyTorch لزيادة سرعة الاستدلال بشكل أكبر. وأخيرًا، تعرف على كيفية استخدام Hugging Face Optimum لتسريع الاستدلال باستخدام ONNX Runtime أو OpenVINO (إذا كنت تستخدم وحدة معالجة مركزية من Intel).

## BetterTransformer

يُسرع BetterTransformer الاستدلال من خلال تنفيذ fastpath (تنفيذ متخصص في PyTorch الأصلي لوظائف Transformer). ويتمثل التحسينان في تنفيذ fastpath فيما يلي:

1. الدمج، الذي يجمع بين عمليات تسلسلية متعددة في "نواة" واحدة لتقليل عدد خطوات الحساب
2. تخطي ندرة التوسيد الفطرية لرموز التوسيد لتجنب الحساب غير الضروري مع المصفوفات المُعششة

كما يحول BetterTransformer جميع عمليات الانتباه إلى استخدام [scaled dot product attention](https://pytorch.org/docs/master/generated/torch.nn.functional.scaled_dot_product_attention) الأكثر كفاءة في الذاكرة.

> ملاحظة: لا يدعم BetterTransformer جميع النماذج. تحقق من هذه [القائمة](https://huggingface.co/docs/optimum/bettertransformer/overview#supported-models) لمعرفة ما إذا كان النموذج يدعم BetterTransformer.

قبل البدء، تأكد من تثبيت Hugging Face Optimum.

قم بتمكين BetterTransformer باستخدام طريقة [`PreTrainedModel.to_bettertransformer`].

```py
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("bigcode/starcoder")
model.to_bettertransformer()
```

## TorchScript

TorchScript هو تمثيل وسيط لنموذج PyTorch يمكن تشغيله في بيئات الإنتاج حيث تكون الأداء مهمة. يمكنك تدريب نموذج في PyTorch ثم تصديره إلى TorchScript لتحرير النموذج من قيود الأداء في Python. تقوم PyTorch بتعقب نموذج لإرجاع [`ScriptFunction`] يتم تحسينه باستخدام التجميع في الوقت المناسب (JIT). مقارنة بوضع التهيئة الافتراضي، عادةً ما يوفر وضع JIT في PyTorch أداءً أفضل للاستدلال باستخدام تقنيات التحسين مثل دمج المشغل.

للاطلاع على مقدمة سهلة إلى TorchScript، راجع البرنامج التعليمي [Introduction to PyTorch TorchScript](https://pytorch.org/tutorials/beginner/Intro_to_TorchScript_tutorial.html).

مع فئة [`Trainer`]`، يمكنك تمكين وضع JIT للاستدلال باستخدام وحدة المعالجة المركزية عن طريق تعيين علم `--jit_mode_eval`:

```bash
python run_qa.py \
--model_name_or_path csarron/bert-base-uncased-squad-v1 \
--dataset_name squad \
--do_eval \
--max_seq_length 384 \
--doc_stride 128 \
--output_dir /tmp/ \
--no_cuda \
--jit_mode_eval
```

> تحذير: بالنسبة لـ PyTorch >= 1.14.0، يمكن أن يستفيد وضع JIT من أي نموذج للتنبؤ والتقييم نظرًا لأن إدخال القاموس مدعوم في `jit.trace`.
> بالنسبة لـ PyTorch < 1.14.0، يمكن أن يفيد وضع JIT نموذجًا إذا تطابق ترتيب معلمات التقديم الخاصة به مع ترتيب إدخال الرباعي في `jit.trace`، مثل نموذج الإجابة على الأسئلة. إذا لم يتطابق ترتيب معلمات التقديم مع ترتيب إدخال الرباعي في `jit.trace`، مثل نموذج تصنيف النص، فسوف يفشل `jit.trace` ونحن نلتقط هذا باستثناء هنا للتراجع. يتم استخدام التسجيل لإخطار المستخدمين.

## تحسين الرسم البياني لـ IPEX

يوفر Intel® Extension for PyTorch (IPEX) مزيدًا من التحسينات في وضع JIT لوحدات معالجة Intel، ونوصي بدمجه مع TorchScript للحصول على أداء أسرع. يقوم تحسين الرسم البياني لـ IPEX بدمج العمليات مثل Multi-head attention، وConcat Linear، وLinear + Add، وLinear + Gelu، وAdd + LayerNorm، والمزيد.

للاستفادة من تحسينات الرسم البياني هذه، تأكد من تثبيت IPEX:

```bash
pip install intel_extension_for_pytorch
```

قم بتعيين علمي `--use_ipex` و`--jit_mode_eval` في فئة [`Trainer`] لتمكين وضع JIT مع تحسينات الرسم البياني:

```bash
python run_qa.py \
--model_name_or_path csarron/bert-base-uncased-squad-v1 \
--dataset_name squad \
--do_eval \
--max_seq_length 384 \
--doc_stride 128 \
--output_dir /tmp/ \
--no_cuda \
--use_ipex \
--jit_mode_eval
```

## Hugging Face Optimum

> تعرف على المزيد من التفاصيل حول استخدام ORT مع Hugging Face Optimum في دليل [Optimum Inference with ONNX Runtime](https://huggingface.co/docs/optimum/onnxruntime/usage_guides/models). يقدم هذا القسم فقط مثالًا موجزًا وبسيطًا.

ONNX Runtime (ORT) هو مسرع نموذج يقوم بتشغيل الاستدلال على وحدات المعالجة المركزية بشكل افتراضي. تدعم Hugging Face Optimum ORT الذي يمكن استخدامه في Hugging Face Transformers، دون إجراء الكثير من التغييرات على التعليمات البرمجية الخاصة بك. تحتاج فقط إلى استبدال Hugging Face Transformers `AutoClass` بما يعادلها [`~optimum.onnxruntime.ORTModel`] للمهمة التي تقوم بحلها، وتحميل نقطة تفتيش بتنسيق ONNX.

على سبيل المثال، إذا كنت تقوم بتشغيل الاستدلال على مهمة الإجابة على الأسئلة، فقم بتحميل نقطة تفتيش [optimum/roberta-base-squad2](https://huggingface.co/optimum/roberta-base-squad2) التي تحتوي على ملف `model.onnx`:

```py
from transformers import AutoTokenizer, pipeline
from optimum.onnxruntime import ORTModelForQuestionAnswering

model = ORTModelForQuestionAnswering.from_pretrained("optimum/roberta-base-squad2")
tokenizer = AutoTokenizer.from_pretrained("deepset/roberta-base-squad2")

onnx_qa = pipeline("question-answering", model=model, tokenizer=tokenizer)

question = "What's my name?"
context = "My name is Philipp and I live in Nuremberg."
pred = onnx_qa(question, context)
```

إذا كان لديك وحدة معالجة مركزية من Intel، فالق نظرة على Hugging Face [Optimum Intel](https://huggingface.co/docs/optimum/intel/index) الذي يدعم مجموعة متنوعة من تقنيات الضغط (الكمية، والتشذيب، وتقطير المعرفة) وأدوات لتحويل النماذج إلى تنسيق [OpenVINO](https://huggingface.co/docs/optimum/intel/inference) للاستدلال عالي الأداء.