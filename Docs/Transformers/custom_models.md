# بناء نماذج مخصصة

تم تصميم مكتبة 🤗 Transformers لتكون قابلة للتوسيع بسهولة. كل نموذج مُشفر بالكامل في مجلد فرعي معين في المستودع دون أي تجريد، لذا يمكنك بسهولة نسخ ملف النمذجة وتعديله وفقًا لاحتياجاتك.

إذا كنت تكتب نموذجًا جديدًا تمامًا، فقد يكون من الأسهل البدء من الصفر. في هذا البرنامج التعليمي، سنُرِيك كيفية كتابة نموذج مخصص وتكوينه حتى يمكن استخدامه داخل Transformers، وكيفية مشاركته مع المجتمع (مع الكود الذي يعتمد عليه) بحيث يمكن لأي شخص استخدامه، حتى إذا لم يكن موجودًا في مكتبة 🤗 Transformers. سنرى كيفية البناء على المحولات وتوسيع الإطار باستخدام hooks وكودك المخصص.

سنقوم بتوضيح كل هذا على نموذج ResNet، عن طريق لف فئة ResNet من مكتبة timm في [`PreTrainedModel`].

# كتابة تكوين مخصص

قبل الغوص في النموذج، دعونا نكتب أولاً تكوينه. تكوين النموذج هو كائن سيحتوي على جميع المعلومات اللازمة لبناء النموذج. كما سنرى في القسم التالي، يمكن للنموذج أن يأخذ فقط `config` ليتم تهيئته، لذا نحتاج حقًا إلى أن يكون هذا الكائن مكتملًا قدر الإمكان.

تتبع النماذج في مكتبة `transformers` نفسها بشكل عام الاتفاقية التي تقبل كائن `config` في طريقة `__init__` الخاصة بها، ثم تمرر كائن `config` بالكامل إلى الطبقات الفرعية في النموذج، بدلاً من كسر كائن التكوين إلى عدة حجج يتم تمريرها جميعها بشكل فردي إلى الطبقات الفرعية. يؤدي كتابة نموذجك بهذه الطريقة إلى كود أبسط مع "مصدر حقيقة" واضح لأي فرط معلمات، كما يسهل إعادة استخدام الكود من النماذج الأخرى في `transformers`.

في مثالنا، سنأخذ بضعة حجج من فئة ResNet التي قد نرغب في ضبطها. ستعطينا التكوينات المختلفة أنواع مختلفة من شبكات ResNet الممكنة. ثم نقوم ببساطة بتخزين تلك الحجج، بعد التحقق من صحة بعضها.

```python
from transformers import PretrainedConfig
from typing import List

class ResnetConfig(PretrainedConfig):
    model_type = "resnet"

    def __init__(
        self,
        block_type="bottleneck",
        layers: List[int] = [3, 4, 6, 3],
        num_classes: int = 1000,
        input_channels: int = 3,
        cardinality: int = 1,
        base_width: int = 64,
        stem_width: int = 64,
        stem_type: str = "",
        avg_down: bool = False,
        **kwargs,
    ):
        if block_type not in ["basic", "bottleneck"]:
            raise ValueError(f"`block_type` must be 'basic' or bottleneck', got {block_type}.")
        if stem_type not in ["", "deep", "deep-tiered"]:
            raise ValueError(f"`stem_type` must be '', 'deep' or 'deep-tiered', got {stem_type}.")

        self.block_type = block_type
        self.layers = layers
        self.num_classes = num_classes
        self.input_channels = input_channels
        self.cardinality = cardinality
        self.base_width = base_width
        self.stem_width = stem_width
        self.stem_type = stem_type
        self.avg_down = avg_down
        super().__init__(**kwargs)
```

الأشياء الثلاثة المهمة التي يجب تذكرها عند كتابة تكوينك الخاص هي ما يلي:

- يجب أن ترث من `PretrainedConfig`،
- يجب أن تقبل `__init__` من `PretrainedConfig` أي kwargs،
- يجب تمرير هذه `kwargs` إلى `__init__` للطبقة العليا.

يضمن الإرث حصولك على جميع الوظائف من مكتبة 🤗 Transformers، في حين أن القيود الأخرى تأتي من حقيقة أن `PretrainedConfig` لديه المزيد من الحقول أكثر من تلك التي تقوم بتعيينها. عند إعادة تحميل تكوين باستخدام طريقة `from_pretrained`، يجب أن يقبل تكوينك هذه الحقول ثم إرسالها إلى الفئة العليا.

تحديد `model_type` لتكوينك (هنا `model_type="resnet"`) ليس إلزاميًا، ما لم ترغب في تسجيل نموذجك باستخدام الفئات التلقائية (راجع القسم الأخير).

مع القيام بذلك، يمكنك بسهولة إنشاء تكوينك وحفظه مثلما تفعل مع أي تكوين نموذج آخر في المكتبة. إليك كيفية إنشاء تكوين resnet50d وحفظه:

```py
resnet50d_config = ResnetConfig(block_type="bottleneck", stem_width=32, stem_type="deep", avg_down=True)
resnet50d_config.save_pretrained("custom-resnet")
```

سيؤدي هذا إلى حفظ ملف باسم `config.json` داخل المجلد `custom-resnet`. بعد ذلك، يمكنك إعادة تحميل تكوينك باستخدام طريقة `from_pretrained`:

```py
resnet50d_config = ResnetConfig.from_pretrained("custom-resnet")
```

يمكنك أيضًا استخدام أي طريقة أخرى من فئة [`PretrainedConfig`]`، مثل [`~PretrainedConfig.push_to_hub`] لتحميل تكوينك مباشرة إلى Hub.

# كتابة نموذج مخصص

الآن بعد أن أصبح لدينا تكوين ResNet، يمكننا المتابعة لكتابة النموذج. في الواقع، سنكتب نموذجين: أحدهما يستخرج الميزات المخفية من دفعة من الصور (مثل [`BertModel`]) والآخر مناسب لتصنيف الصور (مثل [`BertForSequenceClassification`]).

كما ذكرنا سابقًا، سنقوم فقط بلف النموذج لإبقائه بسيطًا في هذا المثال. الشيء الوحيد الذي نحتاج إلى فعله قبل كتابة هذه الفئة هو رسم خريطة بين أنواع الكتل وفئات الكتل الفعلية. بعد ذلك، يتم تعريف النموذج من التكوين عن طريق تمرير كل شيء إلى فئة ResNet:

```py
from transformers import PreTrainedModel
from timm.models.resnet import BasicBlock, Bottleneck, ResNet
from .configuration_resnet import ResnetConfig

BLOCK_MAPPING = {"basic": BasicBlock, "bottleneck": Bottleneck}

class ResnetModel(PreTrainedModel):
    config_class = ResnetConfig

    def __init__(self, config):
        super().__init__(config)
        block_layer = BLOCK_MAPPING[config.block_type]
        self.model = ResNet(
            block_layer,
            config.layers,
            num_classes=config.num_classes,
            in_chans=config.input_channels,
            cardinality=config.cardinality,
            base_width=config.base_width,
            stem_width=config.stem_width,
            stem_type=config.stem_type,
            avg_down=config.avg_down,
        )

    def forward(self, tensor):
        return self.model.forward_features(tensor)
```

بالنسبة للنموذج الذي سيصنف الصور، فإننا نغير فقط طريقة forward:

```py
import torch

class ResnetModelForImageClassification(PreTrainedModel):
    config_class = ResnetConfig

    def __init__(self, config):
        super().__init__(config)
        block_layer = BLOCK_MAPPING[config.block_type]
        self.model = ResNet(
            block_layer,
            config.layers,
            num_classes=config.num_classes,
            in_chans=config.input_channels,
            cardinality=config.cardinality,
            base_width=config.base_width,
            stem_width=config.stem_width,
            stem_type=config.stem_type,
            avg_down=config.avg_down,
        )

    def forward(self, tensor, labels=None):
        logits = self.model(tensor)
        if labels is not None:
            loss = torch.nn.cross_entropy(logits, labels)
            return {"loss": loss, "logits": logits}
        return {"logits": logits}
```

في كلتا الحالتين، لاحظ كيف نرث من `PreTrainedModel` ونستدعي تهيئة الفئة العليا باستخدام `config` (مثلما تفعل عند كتابة وحدة `torch.nn.Module` عادية). السطر الذي يحدد `config_class` ليس إلزاميًا، ما لم ترغب في تسجيل نموذجك باستخدام الفئات التلقائية (راجع القسم الأخير).

إذا كان نموذجك مشابهًا جدًا لنموذج موجود داخل المكتبة، فيمكنك إعادة استخدام نفس التكوين لهذا النموذج.

يمكن لنموذجك أن يعيد أي شيء تريده، ولكن إعادة قاموس كما فعلنا لـ `ResnetModelForImageClassification`، مع تضمين الخسارة عند تمرير العلامات، سيجعل نموذجك قابلًا للاستخدام مباشرة داخل فئة [`Trainer`]. يعد استخدام تنسيق إخراج آخر أمرًا جيدًا طالما أنك تخطط لاستخدام حلقة التدريب الخاصة بك أو مكتبة أخرى للتدريب.

الآن بعد أن أصبح لدينا فئة النموذج، دعنا ننشئ واحدة:

```py
resnet50d = ResnetModelForImageClassification(resnet50d_config)
```

يمكنك استخدام أي من طرق [`PreTrainedModel`]`، مثل [`~PreTrainedModel.save_pretrained`] أو [`~PreTrainedModel.push_to_hub`]. سنستخدم الثاني في القسم التالي، وسنرى كيفية دفع أوزان النموذج مع كود نموذجنا. ولكن أولاً، دعنا نحمل بعض الأوزان المُعلمة مسبقًا داخل نموذجنا.

في حالتك الاستخدامية الخاصة، ستقوم على الأرجح بتدريب نموذجك المخصص على بياناتك الخاصة. للانتقال بسرعة في هذا البرنامج التعليمي، سنستخدم الإصدار المُعلم مسبقًا من resnet50d. نظرًا لأن نموذجنا هو مجرد غلاف حوله، فمن السهل نقل هذه الأوزان:

```py
import timm

pretrained_model = timm.create_model("resnet50d", pretrained=True)
resnet50d.model.load_state_dict(pretrained_model.state_dict())
```

الآن دعونا نرى كيف نتأكد من أنه عند القيام بـ [`~PreTrainedModel.save_pretrained`] أو [`~PreTrainedModel.push_to_hub`]`، يتم حفظ كود النموذج.
بالتأكيد! فيما يلي ترجمة للجزء النصي من التعليمات البرمجية الخاصة بك:

## تسجيل نموذج برمز مخصص للفئات التلقائية

إذا كنت تكتب مكتبة تمدد Hugging Face Transformers، فقد ترغب في تمديد الفئات التلقائية لتشمل نموذجك الخاص. يختلف هذا عن دفع الرمز إلى المركز بمعنى أن المستخدمين سيحتاجون إلى استيراد مكتبتك للحصول على النماذج المخصصة (على عكس تنزيل رمز النموذج تلقائيًا من المركز).

طالما أن تكوينك يحتوي على سمة "model_type" تختلف عن أنواع نماذج Hugging Face Transformers الحالية، وأن فئات النماذج الخاصة بك لديها الصفات "config_class" الصحيحة، يمكنك ببساطة إضافتها إلى الفئات التلقائية على النحو التالي:

لاحظ أن الحجة الأولى المستخدمة عند تسجيل تكوينك المخصص في AutoConfig يجب أن تتطابق مع "model_type" من تكوينك المخصص، والحجة الأولى المستخدمة عند تسجيل نماذجك المخصصة في أي فئة نموذج تلقائي يجب أن تتطابق مع "config_class" لتلك النماذج.

## إرسال الكود إلى المركز

<Tip warning={true}>

هذا API تجريبي وقد يكون به بعض التغييرات الطفيفة في الإصدارات القادمة.

</Tip>

أولاً، تأكد من تحديد نموذجك بالكامل في ملف .py. يمكن أن يعتمد على الواردات النسبية لبعض الملفات الأخرى طالما أن جميع الملفات موجودة في نفس الدليل (لا ندعم الوحدات الفرعية لهذه الميزة حتى الآن). لمثالنا، سنحدد ملفًا يسمى modeling_resnet.py وملف configuration_resnet.py في مجلد دليل العمل الحالي يسمى resnet_model. يحتوي ملف التكوين على رمز ResnetConfig وملف النمذجة يحتوي على رمز ResnetModel وResnetModelForImageClassification.

يحتوي ملف __init__.py على دالة init فارغة، فهو موجود فقط حتى يتمكن بايثون من اكتشاف أن resnet_model يمكن استخدامه كموديول.

<Tip warning={true}>

إذا كنت تقوم بنسخ ملفات النمذجة من المكتبة، فسوف تحتاج إلى استبدال جميع الواردات النسبية في أعلى الملف لاستيرادها من حزمة المحولات.

</Tip>

لاحظ أنه يمكنك إعادة استخدام (أو فئة فرعية) تكوين/نموذج موجود.

لمشاركة نموذجك مع المجتمع، اتبع الخطوات التالية: أولاً استورد نموذج ResNet والتكوين من الملفات التي تم إنشاؤها حديثًا:

ثم عليك أن تخبر المكتبة أنك تريد نسخ ملفات التعليمات البرمجية لتلك الكائنات عند استخدام طريقة save_pretrained وتسجيلها بشكل صحيح باستخدام فئة Auto معينة (خاصة للنماذج)، قم ببساطة بتشغيل:

لاحظ أنه لا توجد حاجة لتحديد فئة تلقائية للتكوين (فهناك فئة تلقائية واحدة فقط لهم، AutoConfig) ولكن الأمر يختلف بالنسبة للنماذج. قد يكون نموذجك المخصص مناسبًا للعديد من المهام المختلفة، لذلك يجب عليك تحديد أي من الفئات التلقائية هو الصحيح لنموذجك.

<Tip>

استخدم register_for_auto_class() إذا كنت تريد نسخ ملفات التعليمات البرمجية. إذا كنت تفضل استخدام الكود الموجود على المركز من مستودع آخر، فلا تحتاج إلى استدعائه. في الحالات التي يوجد فيها أكثر من فئة تلقائية، يمكنك تعديل ملف config.json مباشرةً باستخدام البنية التالية:

</Tip>

الآن، دعنا نقوم بإنشاء التكوين والنماذج كما فعلنا من قبل:

لإرسال النموذج إلى المركز، تأكد من تسجيل الدخول. إما تشغيل في المحطة الطرفية الخاصة بك:

أو من دفتر ملاحظات:

يمكنك بعد ذلك الضغط على مساحة الاسم الخاصة بك (أو منظمة أنت عضو فيها) بهذه الطريقة:

بالإضافة إلى أوزان النمذجة والتكوين بتنسيق JSON، تم أيضًا نسخ ملفات النمذجة والتكوين .py في مجلد "custom-resnet50d" وتحميل النتيجة إلى المركز. يمكنك التحقق من النتيجة في هذا مستودع النموذج.

راجع تعليمات المشاركة لمزيد من المعلومات حول طريقة push إلى Hub.

## استخدام نموذج برمز مخصص

يمكنك استخدام أي تكوين أو نموذج أو محدد مواقع مع ملفات التعليمات البرمجية المخصصة في مستودعها باستخدام الفئات التلقائية وطريقة from_pretrained. يتم فحص جميع الملفات والتعليمات البرمجية التي تم تحميلها إلى المركز بحثًا عن البرامج الضارة (راجع وثائق أمان المركز لمزيد من المعلومات)، ولكن يجب عليك مراجعة رمز النموذج والمؤلف لتجنب تنفيذ التعليمات البرمجية الضارة على جهازك. قم بتعيين trust_remote_code=True لاستخدام نموذج برمز مخصص:

من المستحسن بشدة أيضًا تمرير هاش الالتزام كمراجعة للتأكد من أن مؤلف النماذج لم يقم بتحديث التعليمات البرمجية باستخدام بعض الأسطر الضارة (ما لم تثق تمامًا بمؤلفي النماذج).

لاحظ أنه عند تصفح تاريخ الالتزام لمستودع النموذج على المركز، هناك زر لنسخ هاش الالتزام بسهولة لأي التزام.