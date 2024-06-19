# كيف تنشئ خط أنابيب مخصص؟

في هذا الدليل، سنرى كيف ننشئ خط أنابيب مخصص ونشاركه على [Hub](https://hf.co/models) أو إضافته إلى مكتبة 🤗 Transformers.

أولاً وقبل كل شيء، تحتاج إلى تحديد المدخلات الخام التي سيتمكن خط الأنابيب من معالجتها. يمكن أن تكون هذه المدخلات سلاسل نصية أو بايتات خام أو قواميس أو أي شيء آخر يبدو أنه المدخل المرغوب. حاول أن تبقي هذه المدخلات بسيطة قدر الإمكان، حيث يجعل ذلك التوافق أسهل (حتى عبر لغات أخرى عبر JSON). ستكون هذه المدخلات هي `inputs` لخط الأنابيب (`preprocess`).

بعد ذلك، قم بتعريف `outputs`. اتبع نفس السياسة المطبقة على `inputs`. كلما كانت أبسط، كان ذلك أفضل. ستكون هذه هي المخرجات الخاصة بطريقة `postprocess`.

ابدأ بالوراثة من الفئة الأساسية `Pipeline` مع الطرق الأربع اللازمة لتنفيذ `preprocess`، و`_forward`، و`postprocess`، و`_sanitize_parameters`.

```python
from transformers import Pipeline


class MyPipeline(Pipeline):
    def _sanitize_parameters(self, **kwargs):
        preprocess_kwargs = {}
        if "maybe_arg" in kwargs:
            preprocess_kwargs["maybe_arg"] = kwargs["maybe_arg"]
        return preprocess_kwargs, {}, {}

    def preprocess(self, inputs, maybe_arg=2):
        model_input = Tensor(inputs["input_ids"])
        return {"model_input": model_input}

    def _forward(self, model_inputs):
        # model_inputs == {"model_input": model_input}
        outputs = self.model(**model_inputs)
        # Maybe {"logits": Tensor(...)}
        return outputs

    def postprocess(self, model_outputs):
        best_class = model_outputs["logits"].softmax(-1)
        return best_class
```

يهدف هيكل هذا التفصيل إلى دعم التوافق السلس نسبيًا مع CPU/GPU، مع دعم إجراء المعالجة المسبقة/اللاحقة على CPU على خيوط مختلفة.

ستقوم طريقة `preprocess` بأخذ المدخلات المحددة أصلاً، وتحويلها إلى شيء يمكن إدخاله إلى النموذج. وقد تحتوي على معلومات إضافية وعادة ما تكون على شكل `Dict`.

أما طريقة `_forward` فهي تفصيل للتنفيذ ولا يُقصد بها الاستدعاء المباشر. تعد طريقة `forward` هي الطريقة المفضلة للاستدعاء حيث تحتوي على ضمانات للتأكد من أن كل شيء يعمل على الجهاز المتوقع. إذا كان أي شيء مرتبط بنموذج حقيقي، فيجب أن يكون في طريقة `_forward`، وأي شيء آخر يكون في طريقة preprocess/postprocess.

وستقوم طرق `postprocess` بأخذ مخرجات طريقة `_forward` وتحويلها إلى المخرجات النهائية التي تم تحديدها سابقًا.

أما طريقة `_sanitize_parameters` فهي موجودة للسماح للمستخدمين بتمرير أي معلمات كلما رغبوا في ذلك، سواء كان ذلك أثناء التهيئة `pipeline(...., maybe_arg=4)` أو أثناء الاستدعاء `pipe = pipeline(...); output = pipe(...., maybe_arg=4)`.

وتكون مخرجات طريقة `_sanitize_parameters` هي القواميس الثلاثة للمعلمات التي سيتم تمريرها مباشرة إلى طرق `preprocess`، و`_forward`، و`postprocess`. لا تقم بملء أي شيء إذا لم يستدع المتصل أي معلمة إضافية. يسمح ذلك بالاحتفاظ بالمعلمات الافتراضية في تعريف الدالة وهو ما يكون أكثر "طبيعية".

ومن الأمثلة الكلاسيكية على ذلك معلمة `top_k` في مرحلة ما بعد المعالجة في مهام التصنيف.

```python
>>> pipe = pipeline("my-new-task")
>>> pipe("This is a test")
[{"label": "1-star", "score": 0.8}, {"label": "2-star", "score": 0.1}, {"label": "3-star", "score": 0.05}, {"label": "4-star", "score": 0.025}, {"label": "5-star", "score": 0.025}]

>>> pipe("This is a test", top_k=2)
[{"label": "1-star", "score": 0.8}, {"label": "2-star", "score": 0.1}]
```

ولتحقيق ذلك، سنقوم بتحديث طريقة `postprocess` بمعلمة افتراضية `5`. وتحرير طريقة `_sanitize_parameters` للسماح بهذه المعلمة الجديدة.

```python
def postprocess(self, model_outputs, top_k=5):
    best_class = model_outputs["logits"].softmax(-1)
    # Add logic to handle top_k
    return best_class


def _sanitize_parameters(self, **kwargs):
    preprocess_kwargs = {}
    if "maybe_arg" in kwargs:
        preprocess_kwargs["maybe_arg"] = kwargs["maybe_arg"]

    postprocess_kwargs = {}
    if "top_k" in kwargs:
        postprocess_kwargs["top_k"] = kwargs["top_k"]
    return preprocess_kwargs, {}, postprocess_kwargs
```

حاول أن تبقي المدخلات والمخرجات بسيطة قدر الإمكان، ومن الأفضل أن تكون قابلة للتسلسل باستخدام JSON، حيث يجعل ذلك استخدام خط الأنابيب سهلاً للغاية دون الحاجة إلى أن يفهم المستخدمون أنواعًا جديدة من الكائنات. ومن الشائع أيضًا دعم العديد من أنواع الحجج المختلفة لتسهيل الاستخدام (ملفات الصوت، والتي يمكن أن تكون أسماء ملفات أو عناوين URL أو بايتات خام).

## إضافة المهمة إلى قائمة المهام المدعومة

لتسجيل مهمة `new-task` في قائمة المهام المدعومة، يجب إضافتها إلى `PIPELINE_REGISTRY`:

```python
from transformers.pipelines import PIPELINE_REGISTRY

PIPELINE_REGISTRY.register_pipeline(
    "new-task",
    pipeline_class=MyPipeline,
    pt_model=AutoModelForSequenceClassification,
)
```

يمكنك تحديد نموذج افتراضي إذا أردت، وفي هذه الحالة، يجب أن يأتي مع مراجعة محددة (والتي يمكن أن تكون اسم فرع أو علامة ارتكاز، هنا أخذنا `"abcdef"`) وكذلك النوع:

```python
PIPELINE_REGISTRY.register_pipeline(
    "new-task",
    pipeline_class=MyPipeline,
    pt_model=AutoModelForSequenceClassification,
    default={"pt": ("user/awesome_model", "abcdef")},
    type="text",  # current support type: text, audio, image, multimodal
)
```

## مشاركة خط الأنابيب المخصص على Hub

لمشاركة خط الأنابيب المخصص على Hub، ما عليك سوى حفظ الكود المخصص لفئة `Pipeline` الفرعية في ملف Python. على سبيل المثال، لنفترض أننا نريد استخدام خط أنابيب مخصص لتصنيف أزواج الجمل مثل هذا:

```py
import numpy as np

from transformers import Pipeline


def softmax(outputs):
    maxes = np.max(outputs, axis=-1, keepdims=True)
    shifted_exp = np.exp(outputs - maxes)
    return shifted_exp / shifted_exp.sum(axis=-1, keepdims=True)


class PairClassificationPipeline(Pipeline):
    def _sanitize_parameters(self, **kwargs):
        preprocess_kwargs = {}
        if "second_text" in kwargs:
            preprocess_kwargs["second_text"] = kwargs["second_text"]
        return preprocess_kwargs, {}, {}

    def preprocess(self, text, second_text=None):
        return self.tokenizer(text, text_pair=second_text, return_tensors=self.framework)

    def _forward(self, model_inputs):
        return self.model(**model_inputs)

    def postprocess(self, model_outputs):
        logits = model_outputs.logits[0].numpy()
        probabilities = softmax(logits)

        best_class = np.argmax(probabilities)
        label = self.model.config.id2label[best_class]
        score = probabilities[best_class].item()
        logits = logits.tolist()
        return {"label": label, "score": score, "logits": logits}
```

هذا التنفيذ مستقل عن الإطار، وسيعمل مع نماذج PyTorch وTensorFlow. إذا قمنا بحفظ هذا الكود في ملف باسم `pair_classification.py`، فيمكننا بعد ذلك استيراده وتسجيله على النحو التالي:

```py
from pair_classification import PairClassificationPipeline
from transformers.pipelines import PIPELINE_REGISTRY
from transformers import AutoModelForSequenceClassification, TFAutoModelForSequenceClassification

PIPELINE_REGISTRY.register_pipeline(
    "pair-classification",
    pipeline_class=PairClassificationPipeline,
    pt_model=AutoModelForSequenceClassification,
    tf_model=TFAutoModelForSequenceClassification,
)
```

بمجرد القيام بذلك، يمكننا استخدامه مع نموذج مدرب مسبقًا. على سبيل المثال، تم ضبط النموذج `sgugger/finetuned-bert-mrpc` مسبقًا على مجموعة بيانات MRPC، والتي تصنف أزواج الجمل على أنها صيغ معاد صياغتها أو لا.

```py
from transformers import pipeline

classifier = pipeline("pair-classification", model="sgugger/finetuned-bert-mrpc")
```

بعد ذلك، يمكننا مشاركته على Hub باستخدام طريقة `push_to_hub`:

```py
classifier.push_to_hub("test-dynamic-pipeline")
```

سيقوم هذا بنسخ الملف الذي حددت فيه `PairClassificationPipeline` داخل المجلد `"test-dynamic-pipeline"`، إلى جانب حفظ نموذج ورمز محدد لخط الأنابيب، قبل دفع كل شيء إلى المستودع `{your_username}/test-dynamic-pipeline`. بعد ذلك، يمكن لأي شخص استخدامه طالما قاموا بتوفير الخيار `trust_remote_code=True`:

```py
from transformers import pipeline

classifier = pipeline(model="{your_username}/test-dynamic-pipeline", trust_remote_code=True)
```## إضافة خط الأنابيب إلى تطبيق Transformers

لإضافة خط أنابيب إلى تطبيق Transformers، يجب عليك إضافة وحدة نمطية جديدة في الوحدة الفرعية "pipelines" مع كود خط الأنابيب الخاص بك، ثم إضافته إلى قائمة المهام المحددة في "pipelines/__init__.py".

بعد ذلك، ستحتاج إلى إضافة اختبارات. قم بإنشاء ملف جديد "tests/test_pipelines_MY_PIPELINE.py" مع أمثلة على الاختبارات الأخرى. ستكون وظيفة "run_pipeline_test" عامة جدًا وتعمل على نماذج صغيرة عشوائية على كل بنية ممكنة كما هو محدد بواسطة "model_mapping" و"tf_model_mapping".

من المهم جدًا اختبار التوافق المستقبلي، مما يعني أنه إذا قام شخص ما بإضافة نموذج جديد لـ "XXXForQuestionAnswering"، فسيحاول اختبار خط الأنابيب تشغيله. نظرًا لأن النماذج عشوائية، فمن المستحيل التحقق من القيم الفعلية، ولهذا يوجد مساعد "ANY" الذي سيحاول ببساطة مطابقة إخراج نوع خط الأنابيب.

يجب عليك أيضًا تنفيذ اختبارين (من الناحية المثالية 4).

- `test_small_model_pt`: قم بتعريف نموذج صغير واحد لهذا الخط الأنابيب (لا يهم إذا كانت النتائج غير منطقية) واختبار مخرجات خط الأنابيب. يجب أن تكون النتائج نفسها كما في "test_small_model_tf".

- `test_small_model_tf`: قم بتعريف نموذج صغير واحد لهذا الخط الأنابيب (لا يهم إذا كانت النتائج غير منطقية) واختبار مخرجات خط الأنابيب. يجب أن تكون النتائج نفسها كما في "test_small_model_pt".

- `test_large_model_pt` (اختياري): يقوم باختبار خط الأنابيب على خط أنابيب حقيقي من المفترض أن تكون النتائج منطقية. هذه الاختبارات بطيئة ويجب تمييزها على هذا النحو. هنا، الهدف هو عرض خط الأنابيب والتأكد من عدم وجود انحراف في الإصدارات المستقبلية.

- `test_large_model_tf` (اختياري): يقوم باختبار خط الأنابيب على خط أنابيب حقيقي من المفترض أن تكون النتائج منطقية. هذه الاختبارات بطيئة ويجب تمييزها على هذا النحو. هنا، الهدف هو عرض خط الأنابيب والتأكد من عدم وجود انحراف في الإصدارات المستقبلية.