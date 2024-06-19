# التصدير إلى ONNX

غالباً ما يتطلب نشر نماذج 🤗 Transformers في بيئات الإنتاج، أو يمكن أن يستفيد من تصدير النماذج إلى تنسيق مسلسل يمكن تحميله وتشغيله على أجهزة وبرامج متخصصة.

🤗 Optimum هو امتداد لـ Transformers يمكّن تصدير النماذج من PyTorch أو TensorFlow إلى تنسيقات مسلسلة مثل ONNX و TFLite من خلال وحدة "exporters". يوفر 🤗 Optimum أيضًا مجموعة من أدوات تحسين الأداء لتدريب النماذج وتشغيلها على أجهزة مستهدفة بكفاءة قصوى.

يوضح هذا الدليل كيفية تصدير نماذج 🤗 Transformers إلى ONNX باستخدام 🤗 Optimum، وللحصول على الدليل الخاص بتصدير النماذج إلى TFLite، يرجى الرجوع إلى صفحة [التصدير إلى TFLite](tflite).

## التصدير إلى ONNX

[ONNX (Open Neural Network Exchange)](http://onnx.ai) هو معيار مفتوح يحدد مجموعة مشتركة من العمليات وتنسيق ملف مشترك لتمثيل نماذج التعلم العميق في مجموعة متنوعة من الأطر، بما في ذلك PyTorch و TensorFlow. عندما يتم تصدير نموذج إلى تنسيق ONNX، يتم استخدام هذه العمليات لبناء رسم بياني حسابي (يُطلق عليه غالبًا اسم _تمثيل وسيط_) يمثل تدفق البيانات عبر الشبكة العصبية.

من خلال عرض رسم بياني بمشغلين وأنواع بيانات موحدة، تجعل ONNX من السهل التبديل بين الأطر. على سبيل المثال، يمكن تصدير نموذج مدرب في PyTorch إلى تنسيق ONNX ثم استيراده في TensorFlow (والعكس صحيح).

بمجرد تصديره إلى تنسيق ONNX، يمكن تحسين النموذج للتنبؤ باستخدام تقنيات مثل [تحسين الرسم البياني](https://huggingface.co/docs/optimum/onnxruntime/usage_guides/optimization) و [التكميم](https://huggingface.co/docs/optimum/onnxruntime/usage_guides/quantization).

- يمكن تشغيله باستخدام ONNX Runtime عبر فئات [`ORTModelForXXX`](https://huggingface.co/docs/optimum/onnxruntime/package_reference/modeling_ort)، والتي تتبع نفس واجهة برمجة التطبيقات "AutoModel" التي اعتدت عليها في 🤗 Transformers.

- يمكن تشغيله باستخدام [خطوط أنابيب الاستدلال المحسنة](https://huggingface.co/docs/optimum/main/en/onnxruntime/usage_guides/pipelines)، والتي لها نفس واجهة برمجة التطبيقات مثل وظيفة [`pipeline`] في 🤗 Transformers.

يوفر 🤗 Optimum الدعم لتصدير ONNX من خلال الاستفادة من كائنات التكوين. تأتي كائنات التكوين جاهزة لعدد من هندسات النماذج، وهي مصممة لتكون قابلة للتوسيع بسهولة إلى هندسات أخرى.

لعرض قائمة بالتكوينات الجاهزة، يرجى الرجوع إلى [وثائق 🤗 Optimum](https://huggingface.co/docs/optimum/exporters/onnx/overview).

هناك طريقتان لتصدير نموذج 🤗 Transformers إلى ONNX، نوضح كلا الطريقتين فيما يلي:

- التصدير باستخدام 🤗 Optimum عبر سطر الأوامر.
- التصدير باستخدام 🤗 Optimum مع `optimum.onnxruntime`.

### تصدير نموذج 🤗 Transformers إلى ONNX باستخدام سطر الأوامر

لتصدير نموذج 🤗 Transformers إلى ONNX، قم أولاً بتثبيت اعتماد إضافي:

```bash
pip install optimum[exporters]
```

للاطلاع على جميع الحجج المتاحة، راجع [وثائق 🤗 Optimum](https://huggingface.co/docs/optimum/exporters/onnx/usage_guides/export_a_model#exporting-a-model-to-onnx-using-the-cli)، أو عرض المساعدة في سطر الأوامر:

```bash
optimum-cli export onnx --help
```

لتصدير نقطة تفتيش نموذج من 🤗 Hub، على سبيل المثال، `distilbert/distilbert-base-uncased-distilled-squad`، قم بتشغيل الأمر التالي:

```bash
optimum-cli export onnx --model distilbert/distilbert-base-uncased-distilled-squad distilbert_base_uncased_squad_onnx/
```

يجب أن تشاهد السجلات التي تشير إلى التقدم المحرز وتظهر المكان الذي تم فيه حفظ ملف `model.onnx` الناتج، مثل هذا:

```bash
Validating ONNX model distilbert_base_uncased_squad_onnx/model.onnx...
-[✓] ONNX model output names match reference model (start_logits, end_logits)
- Validating ONNX Model output "start_logits":
-[✓] (2, 16) matches (2, 16)
-[✓] all values close (atol: 0.0001)
- Validating ONNX Model output "end_logits":
-[✓] (2, 16) matches (2, 16)
-[✓] all values close (atol: 0.0001)
The ONNX export succeeded and the exported model was saved at: distilbert_base_uncased_squad_onnx
```

يوضح المثال أعلاه تصدير نقطة تفتيش من 🤗 Hub. عند تصدير نموذج محلي، تأكد أولاً من حفظ ملفات أوزان النموذج ومحول الرموز في نفس الدليل (`local_path`). عند استخدام سطر الأوامر، قم بتمرير `local_path` إلى وسيط `model` بدلاً من اسم نقطة التفتيش على 🤗 Hub وقدم وسيط `--task`.

يمكنك مراجعة قائمة المهام المدعومة في [وثائق 🤗 Optimum](https://huggingface.co/docs/optimum/exporters/task_manager).

إذا لم يتم توفير وسيط `task`، فسيتم تعيينه افتراضيًا إلى هندسة النموذج دون أي رأس محدد للمهمة.

```bash
optimum-cli export onnx --model local_path --task question-answering distilbert_base_uncased_squad_onnx/
```

يمكن بعد ذلك تشغيل ملف `model.onnx` الناتج على أحد [المسرعات](https://onnx.ai/supported-tools.html#deployModel) العديدة التي تدعم معيار ONNX. على سبيل المثال، يمكننا تحميل النموذج وتشغيله باستخدام [ONNX Runtime](https://onnxruntime.ai/) كما يلي:

```python
>>> from transformers import AutoTokenizer
>>> from optimum.onnxruntime import ORTModelForQuestionAnswering

>>> tokenizer = AutoTokenizer.from_pretrained("distilbert_base_uncased_squad_onnx")
>>> model = ORTModelForQuestionAnswering.from_pretrained("distilbert_base_uncased_squad_onnx")
>>> inputs = tokenizer("What am I using?", "Using DistilBERT with ONNX Runtime!", return_tensors="pt")
>>> outputs = model(**inputs)
```

تكون العملية مماثلة بالنسبة إلى نقاط تفتيش TensorFlow على Hub. على سبيل المثال، إليك كيفية تصدير نقطة تفتيش TensorFlow نقية من [منظمة Keras](https://huggingface.co/keras-io):

```bash
optimum-cli export onnx --model keras-io/transformers-qa distilbert_base_cased_squad_onnx/
```

### تصدير نموذج 🤗 Transformers إلى ONNX باستخدام `optimum.onnxruntime`

بدلاً من استخدام سطر الأوامر، يمكنك تصدير نموذج 🤗 Transformers إلى ONNX برمجيًا على النحو التالي:

```python
>>> from optimum.onnxruntime import ORTModelForSequenceClassification
>>> from transformers import AutoTokenizer

>>> model_checkpoint = "distilbert_base_uncased_squad"
>>> save_directory = "onnx/"

>>> # Load a model from transformers and export it to ONNX
>>> ort_model = ORTModelForSequenceClassification.from_pretrained(model_checkpoint, export=True)
>>> tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

>>> # Save the onnx model and tokenizer
>>> ort_model.save_pretrained(save_directory)
>>> tokenizer.save_pretrained(save_directory)
```

### تصدير نموذج لهندسة غير مدعومة

إذا كنت ترغب في المساهمة بإضافة دعم لنموذج لا يمكن تصديره حاليًا، فيجب عليك أولاً التحقق مما إذا كان مدعومًا في [`optimum.exporters.onnx`](https://huggingface.co/docs/optimum/exporters/onnx/overview)، وإذا لم يكن الأمر كذلك، [قم بالمساهمة في 🤗 Optimum](https://huggingface.co/docs/optimum/exporters/onnx/usage_guides/contribute) مباشرة.

### تصدير نموذج باستخدام `transformers.onnx`

<Tip warning={true}>
لم تعد `tranformers.onnx` مدعومة، يرجى تصدير النماذج باستخدام 🤗 Optimum كما هو موضح أعلاه. سيتم إزالة هذا القسم في الإصدارات المستقبلية.
</Tip>

لتصدير نموذج 🤗 Transformers إلى ONNX باستخدام `tranformers.onnx`، قم بتثبيت الاعتماديات الإضافية:

```bash
pip install transformers[onnx]
```

استخدم حزمة `transformers.onnx` كنموذج Python لتصدير نقطة تفتيش باستخدام تكوين جاهز:

```bash
python -m transformers.onnx --model=distilbert/distilbert-base-uncased onnx/
```

هذا يصدر رسم بياني ONNX لنقطة التفتيش التي حددها وسيط `--model`. قم بتمرير أي نقطة تفتيش على 🤗 Hub أو واحدة مخزنة محليًا.

يمكن بعد ذلك تشغيل ملف `model.onnx` الناتج على أحد المسرعات العديدة التي تدعم معيار ONNX. على سبيل المثال، قم بتحميل النموذج وتشغيله باستخدام ONNX Runtime كما يلي:

```python
>>> from transformers import AutoTokenizer
>>> from onnxruntime import InferenceSession

>>> tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")
>>> session = InferenceSession("onnx/model.onnx")
>>> # ONNX Runtime expects NumPy arrays as input
>>> inputs = tokenizer("Using DistilBERT with ONNX Runtime!", return_tensors="np")
>>> outputs = session.run(output_names=["last_hidden_state"], input_feed=dict(inputs))
```

يمكن الحصول على أسماء الإخراج المطلوبة (مثل `["last_hidden_state"]`) عن طريق إلقاء نظرة على تكوين ONNX لكل نموذج. على سبيل المثال، بالنسبة لـ DistilBERT، لدينا:

```python
>>> from transformers.models.distilbert import DistilBertConfig, DistilBertOnnxConfig

>>> config = DistilBertConfig()
>>> onnx_config = DistilBertOnnxConfig(config)
>>> print(list(onnx_config.outputs.keys()))
["last_hidden_state"]
```

تكون العملية مماثلة بالنسبة إلى نقاط تفتيش TensorFlow على Hub. على سبيل المثال، قم بتصدير نقطة تفتيش TensorFlow نقية على النحو التالي:

```bash
python -m transformers.onnx --model=keras-io/transformers-qa onnx/
```

لتصدير نموذج مخزن محليًا، قم بحفظ ملفات أوزان النموذج ومحول الرموز في نفس الدليل (مثل `local-pt-checkpoint`)، ثم قم بتصديره إلى ONNX عن طريق توجيه وسيط `--model` لحزمة `transformers.onnx` إلى الدليل المطلوب:

```bash
python -m transformers.onnx --model=local-pt-checkpoint onnx/
```