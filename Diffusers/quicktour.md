[[open-in-colab]]
 # جولة سريعة 

تُدرب نماذج الانتشار على إزالة التشويش من الضوضاء الغاوسية بشكل تدريجي لتوليد عينة ذات اهتمام، مثل صورة أو صوت. وقد أثار هذا اهتمامًا هائلاً بالذكاء الاصطناعي التوليدي، وربما تكون قد رأيت أمثلة على الصور المولدة بالانتشار على الإنترنت. 🧨 Diffusers هي مكتبة تهدف إلى جعل نماذج الانتشار في متناول الجميع.

سواء كنت مطورًا أو مستخدمًا عاديًا، ستتعرفك هذه الجولة السريعة على 🧨 Diffusers وستساعدك على البدء في التوليد بسرعة! هناك ثلاثة مكونات رئيسية للمكتبة يجب معرفتها:

* [`DiffusionPipeline`] هي فئة عالية المستوى مصممة لتوليد عينات بسرعة من نماذج الانتشار المُدربة مسبقًا للاستدلال.
* النماذج المُدربة مسبقًا [model](./api/models) الشهيرة ووحدات البناء التي يمكن استخدامها لإنشاء أنظمة الانتشار.
* العديد من [schedulers](./api/schedulers/overview) المختلفة - خوارزميات تتحكم في كيفية إضافة الضوضاء للتدريب، وكيفية توليد الصور الخالية من التشويش أثناء الاستدلال.

ستُظهر لك هذه الجولة السريعة كيفية استخدام [`DiffusionPipeline`] للاستدلال، ثم تشرح لك كيفية دمج نموذج ومخطط زمني لتكرار ما يحدث داخل [`DiffusionPipeline`].

<Tip>

هذه الجولة السريعة هي نسخة مبسطة من الدفتر 🧨 Diffusers [notebook](https://colab.research.google.com/github.com/huggingface/notebooks/blob/main/diffusers/diffusers_intro.ipynb) لمساعدتك على البدء بسرعة. إذا كنت ترغب في معرفة المزيد عن هدف 🧨 Diffusers وفلسفة التصميم وتفاصيل إضافية حول واجهة برمجة التطبيقات الأساسية الخاصة بها، فراجع الدفتر!

</Tip>

قبل البدء، تأكد من تثبيت جميع المكتبات الضرورية:

```py
# قم بإلغاء التعليق لتثبيت المكتبات الضرورية في Colab
#! pip install --upgrade diffusers accelerate transformers
```

- [🤗 Accelerate](https://huggingface.co/docs/accelerate/index) يسرع تحميل النموذج للاستدلال والتدريب.
- [🤗 Transformers](https://huggingface.co/docs/transformers/index) مطلوب لتشغيل أكثر نماذج الانتشار شهرة، مثل [Stable Diffusion](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/overview).

## DiffusionPipeline

[`DiffusionPipeline`] هي أسهل طريقة لاستخدام نظام انتشار مُدرب مسبقًا للاستدلال. وهو نظام شامل يحتوي على النموذج والمخطط الزمني. يمكنك استخدام [`DiffusionPipeline`] جاهزًا للعديد من المهام. الق نظرة على الجدول أدناه لبعض المهام المدعومة، وللحصول على قائمة كاملة بالمهام المدعومة، راجع جدول [🧨 Diffusers Summary](./api/pipelines/overview#diffusers-summary).

| المهمة | الوصف | الأنابيب |
|------------------------------|--------------------------------------------------------------------------------------------------------------|-----------------|
| Unconditional Image Generation | قم بتوليد صورة من الضوضاء الغاوسية | [unconditional_image_generation](./using-diffusers/unconditional_image_generation) |
| Text-Guided Image Generation | قم بتوليد صورة بناءً على موجه نصي | [conditional_image_generation](./using-diffusers/conditional_image_generation) |
| Text-Guided Image-to-Image Translation | تكييف صورة موجهة بنص موجه | [img2img](./using-diffusers/img2img) |
| Text-Guided Image-Inpainting | املأ الجزء المُقنع من الصورة بالنظر إلى الصورة والقناع وموجه النص | [inpaint](./using-diffusers/inpaint) |
| Text-Guided Depth-to-Image Translation | تكييف أجزاء من صورة موجهة بنص مع الحفاظ على البنية من خلال تقدير العمق | [depth2img](./using-diffusers/depth2img) |

ابدأ بإنشاء مثيل من [`DiffusionPipeline`] وحدد نقطة تفتيش الأنابيب التي تريد تنزيلها.
يمكنك استخدام [`DiffusionPipeline`] لأي [checkpoint](https://huggingface.co/models?library=diffusers&sort=downloads) المخزن على Hugging Face Hub.
في هذه الجولة السريعة، ستقوم بتحميل نقطة تفتيش [`stable-diffusion-v1-5`](https://huggingface.co/runwayml/stable-diffusion-v1-5) لتوليد الصور بناءً على النص.

<Tip warning={true}>

بالنسبة لنماذج [Stable Diffusion](https://huggingface.co/CompVis/stable-diffusion)، يرجى قراءة [الرخصة](https://huggingface.co/spaces/CompVis/stable-diffusion-license) بعناية قبل تشغيل النموذج. 🧨 Diffusers تنفذ [`safety_checker`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/safety_checker.py) لمنع المحتوى المسيء أو الضار، ولكن قدرات النموذج المحسنة لتوليد الصور يمكن أن تنتج محتوى ضارًا محتملاً.

</Tip>

قم بتحميل النموذج باستخدام طريقة [`~DiffusionPipeline.from_pretrained`]:

```python
>>> from diffusers import DiffusionPipeline

>>> pipeline = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", use_safetensors=True)
```

يقوم [`DiffusionPipeline`] بتنزيل وتخزين جميع مكونات النمذجة والتمثيل والجدولة. ستلاحظ أن خط أنابيب Stable Diffusion يتكون من [`UNet2DConditionModel`] و [`PNDMScheduler`]، من بين أشياء أخرى:

```py
>>> pipeline
StableDiffusionPipeline {
"_class_name": "StableDiffusionPipeline"،
"_diffusers_version": "0.21.4"،
...،
"scheduler": [
"diffusers"،
"PNDMScheduler"
]،
...،
"unet": [
"diffusers"،
"UNet2DConditionModel"
]،
"vae": [
"diffusers"،
"AutoencoderKL"
]
}
```

نوصي بشدة بتشغيل الأنبوب على وحدة معالجة الرسومات (GPU) لأن النموذج يتكون من حوالي 1.4 مليار معلمة.
يمكنك نقل كائن المولد إلى وحدة معالجة الرسومات (GPU)، تمامًا كما تفعل في PyTorch:

```python
>>> pipeline.to("cuda")
```

الآن يمكنك تمرير موجه نصي إلى `pipeline` لتوليد صورة، ثم الوصول إلى الصورة الخالية من التشويش. بشكل افتراضي، يتم لف إخراج الصورة في كائن [`PIL.Image`](https://pillow.readthedocs.io/en/stable/reference/Image.html?highlight=image#the-image-class).

```python
>>> image = pipeline("An image of a squirrel in Picasso style").images[0]
>>> image
```

<div class="flex justify-center">
<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/image_of_squirrel_painting.png"/>
</div>

احفظ الصورة عن طريق استدعاء `save`:

```python
>>> image.save("image_of_squirrel_painting.png")
```

### خط أنابيب محلي

يمكنك أيضًا استخدام خط الأنابيب محليًا. الفرق الوحيد هو أنك بحاجة إلى تنزيل الأوزان أولاً:

```bash
!git lfs install
!git clone https://huggingface.co/runwayml/stable-diffusion-v1-5
```

ثم قم بتحميل الأوزان المحفوظة في خط الأنابيب:

```python
>>> pipeline = DiffusionPipeline.from_pretrained("./stable-diffusion-v1-5"، use_safetensors=True)
```

الآن، يمكنك تشغيل خط الأنابيب كما هو موضح في القسم أعلاه.

### تبديل المخططات الزمنية

تأتي المخططات الزمنية المختلفة بسرعات مختلفة لإزالة التشويش ومفاضلات الجودة. أفضل طريقة لمعرفة أي منها يعمل بشكل أفضل بالنسبة لك هي تجربتها! إحدى الميزات الرئيسية لـ 🧨 Diffusers هي السماح لك بالتبديل بسهولة بين المخططات الزمنية. على سبيل المثال، لاستبدال [`PNDMScheduler`] الافتراضي بـ [`EulerDiscreteScheduler`]، قم بتحميله باستخدام طريقة [`~diffusers.ConfigMixin.from_config`]:

```py
>>> from diffusers import EulerDiscreteScheduler

>>> pipeline = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5"، use_safetensors=True)
>>> pipeline.scheduler = EulerDiscreteScheduler.from_config(pipeline.scheduler.config)
```

جرب توليد صورة بالمخطط الزمني الجديد وانظر إذا لاحظت أي فرق!

في القسم التالي، سنلقي نظرة فاحصة على المكونات - النموذج والمخطط الزمني - التي تشكل [`DiffusionPipeline`] وتعلم كيفية استخدام هذه المكونات لتوليد صورة لقطة.
## النماذج

تأخذ معظم النماذج عينة مشوشة، وفي كل خطوة زمنية، تتنبأ بـ "بقايا التشويش" (تتعلم النماذج الأخرى التنبؤ بالعينة السابقة مباشرة أو السرعة أو ["v-prediction"])، وهو الفرق بين صورة أقل تشويشًا والصورة المدخلة. يمكنك مزج ومطابقة النماذج لإنشاء أنظمة انتشار أخرى.

يتم بدء النماذج باستخدام طريقة [`~ModelMixin.from_pretrained`] التي تقوم أيضًا بتخزين الأوزان النموذجية محليًا بحيث تكون أسرع في المرة التالية التي تقوم فيها بتحميل النموذج. للحصول على جولة سريعة، ستقوم بتحميل [`UNet2DModel`]، وهو نموذج أساسي لتوليد الصور غير المشروط مع نقطة تفتيش مدربة على صور القطط:

```py
>>> from diffusers import UNet2DModel

>>> repo_id = "google/ddpm-cat-256"
>>> model = UNet2DModel.from_pretrained(repo_id, use_safetensors=True)
```

للاطلاع على معلمات النموذج، اتصل بـ `model.config`:

```py
>>> model.config
```

تكوين النموذج هو عبارة عن قاموس مجمد، مما يعني أنه لا يمكن تغيير هذه المعلمات بعد إنشاء النموذج. هذا مقصود ويضمن أن المعلمات المستخدمة لتحديد بنية النموذج في البداية تظل كما هي، في حين يمكن ضبط المعلمات الأخرى أثناء الاستدلال.

بعض أهم المعلمات هي:

* `sample_size`: البعد العمودي والأفقي للعينة المدخلة.
* `in_channels`: عدد قنوات الإدخال للعينة المدخلة.
* `down_block_types` و `up_block_types`: نوع كتل التصغير والتصغير المستخدمة لإنشاء بنية UNet.
* `block_out_channels`: عدد قنوات الإخراج لكتل التصغير؛ تستخدم أيضًا بترتيب عكسي لعدد قنوات الإدخال لكتل التصغير.
* `layers_per_block`: عدد كتل ResNet الموجودة في كل كتلة UNet.

لاستخدام النموذج للاستدلال، قم بإنشاء شكل الصورة مع ضوضاء غاوسية عشوائية. يجب أن يكون لها محور "دفعة" لأن النموذج يمكنه استقبال ضوضاء عشوائية متعددة، ومحور "قناة" مطابق لعدد قنوات الإدخال، ومحور "sample_size" لارتفاع الصورة وعرضها:

```py
>>> import torch

>>> torch.manual_seed(0)

>>> noisy_sample = torch.randn(1, model.config.in_channels, model.config.sample_size, model.config.sample_size)
>>> noisy_sample.shape
torch.Size([1, 3, 256, 256])
```

للاستدلال، قم بتمرير الصورة المشوشة و "خطوة زمنية" إلى النموذج. يشير "الخطوة الزمنية" إلى مدى تشويش صورة الإدخال، مع وجود تشويش أكبر في البداية وأقل في النهاية. يساعد هذا النموذج على تحديد موضعه في عملية الانتشار، سواء كان أقرب إلى البداية أو النهاية. استخدم طريقة "sample" للحصول على إخراج النموذج:

```py
>>> with torch.no_grad():
...     noisy_residual = model(sample=noisy_sample, timestep=2).sample
```

ومع ذلك، لإنشاء أمثلة فعلية، ستحتاج إلى جدول زمني لتوجيه عملية إزالة التشويش. في القسم التالي، ستتعلم كيفية ربط نموذج بجدول زمني.

## الجداول الزمنية

تدير الجداول الزمنية الانتقال من عينة مشوشة إلى عينة أقل تشويشًا بالنظر إلى إخراج النموذج - في هذه الحالة، يكون "noisy_residual".

🧨 Diffusers هي مجموعة أدوات لإنشاء أنظمة الانتشار. في حين أن [`DiffusionPipeline`] هي طريقة ملائمة للبدء بنظام انتشار مسبق البناء، يمكنك أيضًا اختيار مكونات النموذج والجدول الزمني الخاصة بك بشكل منفصل لإنشاء نظام انتشار مخصص.

للحصول على جولة سريعة، ستقوم بتنفيذ مثيل [`DDPMScheduler`] باستخدام طريقة [`~diffusers.ConfigMixin.from_config`]:

```py
>>> from diffusers import DDPMScheduler

>>> scheduler = DDPMScheduler.from_pretrained(repo_id)
>>> scheduler
DDPMScheduler {
"_class_name": "DDPMScheduler"،
"_diffusers_version": "0.21.4"،
"beta_end": 0.02،
"beta_schedule": "linear"،
"beta_start": 0.0001،
"clip_sample": true،
"clip_sample_range": 1.0،
"dynamic_thresholding_ratio": 0.995،
"num_train_timesteps": 1000،
"prediction_type": "epsilon"،
"sample_max_value": 1.0،
"steps_offset": 0،
"thresholding": false،
"timestep_spacing": "leading"،
"trained_betas": null،
"variance_type": "fixed_small"
}
```

💡 على عكس النموذج، لا يحتوي الجدول الزمني على أوزان قابلة للتدريب وهو خالٍ من المعلمات!

بعض أهم المعلمات هي:

* `num_train_timesteps`: طول عملية إزالة التشويش، أو بعبارة أخرى، عدد الخطوات الزمنية اللازمة لمعالجة الضوضاء العشوائية غاوسية إلى عينة بيانات.
* `beta_schedule`: نوع جدول التشويش المستخدم للاستدلال والتدريب.
* `beta_start` و `beta_end`: قيم التشويش الابتدائية والنهائية لجدول التشويش.

للتنبؤ بصورة أقل تشويشًا قليلاً، قم بتمرير ما يلي إلى طريقة [`~diffusers.DDPMScheduler.step`] في الجدول الزمني: إخراج النموذج، و "الخطوة الزمنية"، و "العينة" الحالية.

```py
>>> less_noisy_sample = scheduler.step(model_output=noisy_residual, timestep=2, sample=noisy_sample).prev_sample
>>> less_noisy_sample.shape
torch.Size([1, 3, 256, 256])
```

يمكن تمرير `less_noisy_sample` إلى الخطوة الزمنية التالية حيث ستصبح أقل تشويشًا! دعنا نجمع كل شيء الآن ونصور عملية إزالة التشويش بأكملها.

أولاً، قم بإنشاء دالة تقوم بمعالجة الصورة المزالة التشويش وعرضها كـ `PIL.Image`:

```py
>>> import PIL.Image
>>> import numpy as np


>>> def display_sample(sample, i):
...     image_processed = sample.cpu().permute(0, 2, 3, 1)
...     image_processed = (image_processed + 1.0) * 127.5
...     image_processed = image_processed.numpy().astype(np.uint8)

...     image_pil = PIL.Image.fromarray(image_processed[0])
...     display(f"Image at step {i}")
...     display(image_pil)
```

للتسريع عملية إزالة التشويش، قم بنقل الإدخال والنموذج إلى وحدة معالجة الرسومات (GPU):

```py
>>> model.to("cuda")
>>> noisy_sample = noisy_sample.to("cuda")
```

الآن قم بإنشاء حلقة إزالة التشويش التي تتنبأ ببقايا العينة الأقل تشويشًا، وتحسب العينة الأقل تشويشًا باستخدام الجدول الزمني:

```py
>>> import tqdm

>>> sample = noisy_sample

>>> for i, t in enumerate(tqdm.tqdm(scheduler.timesteps)):
...     # 1. predict noise residual
...     with torch.no_grad():
...         residual = model(sample, t).sample

...     # 2. compute less noisy image and set x_t -> x_t-1
...     sample = scheduler.step(residual, t, sample).prev_sample

...     # 3. optionally look at image
...     if (i + 1) % 50 == 0:
...         display_sample(sample, i + 1)
```

استرخي واستمتع بمشاهدة صورة قطة يتم إنشاؤها من لا شيء سوى الضوضاء! 😻

<div class="flex justify-center">
<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/diffusion-quicktour.png"/>
</div>

## الخطوات التالية

من المؤمل أن تكون قد أنشأت بعض الصور الرائعة باستخدام 🧨 Diffusers في هذه الجولة السريعة! بالنسبة لخطواتك التالية، يمكنك:

* تدريب أو ضبط نموذج لتوليد صورك الخاصة في [التدريب](./tutorials/basic_training) التعليمي.
* راجع النصوص البرمجية الرسمية والمجتمعية [لتدريب أو ضبط النصوص](https://github.com/huggingface/diffusers/tree/main/examples#-diffusers-examples) لمجموعة متنوعة من حالات الاستخدام.
* تعرف على المزيد حول تحميل الجداول الزمنية والوصول إليها وتغييرها ومقارنتها في دليل [Using different Schedulers](./using-diffusers/schedulers).
* استكشف هندسة المطالبات، والتحسينات المتعلقة بالسرعة والذاكرة، والنصائح والحيل لإنشاء صور عالية الجودة باستخدام دليل [Stable Diffusion](./stable_diffusion).
* تعمق في تسريع 🧨 Diffusers باستخدام أدلة حول [Optimized PyTorch on a GPU](./optimization/fp16)، وأدلة الاستدلال لـ [Stable Diffusion on Apple Silicon (M1/M2)](./optimization/mps) و [ONNX Runtime](./optimization/onnx).