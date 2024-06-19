# ميزات الجدولة

يعد الجدولة مكونًا مهمًا في أي نموذج انتشار لأنه يتحكم في عملية إزالة التشويش (أو المعايرة) بأكملها. هناك العديد من أنواع الجداول، حيث يتم تحسين بعضها للسرعة والبعض الآخر للجودة. باستخدام أجهزة الانتشار، يمكنك تعديل تكوين الجدولة لاستخدام جداول التشويش المخصصة، والقيم الانحرافات المعيارية، وإعادة تحجيم جدول التشويش. يمكن أن يكون لتغيير هذه المعلمات تأثير عميق على جودة الاستدلال وسرعته.

سيوضح هذا الدليل كيفية استخدام هذه الميزات لتحسين جودة الاستدلال.

> [!TIP]
> تدعم أجهزة الانتشار حاليًا معلمات "timesteps" و "sigmas" لقائمة مختارة من الجداول وأنابيب. لا تتردد في فتح [طلب ميزة](https://github.com/huggingface/diffusers/issues/new/choose) إذا كنت تريد توسيع هذه المعلمات إلى جدول وأنبوب لا يدعمهما حاليًا!

## جداول الخطوة الزمنية

تحدد الخطوة الزمنية أو جدول التشويش مقدار التشويش في كل خطوة معايرة. يستخدم الجدولة هذا لإنشاء صورة بالكمية المقابلة من التشويش في كل خطوة. يتم إنشاء جدول الخطوة الزمنية من التكوين الافتراضي للجدولة، ولكن يمكنك تخصيص الجدولة لاستخدام جداول المعايرة الجديدة والمحسنة التي لم يتم تضمينها في أجهزة الانتشار بعد.

على سبيل المثال، تعد [محاذاة خطواتك (AYS)](https://research.nvidia.com/labs/toronto-ai/AlignYourSteps/) طريقة لتحسين جدول المعايرة لإنشاء صورة عالية الجودة في غضون 10 خطوات فقط. الجدول الأمثل [10 خطوات](https://github.com/huggingface/diffusers/blob/a7bf77fc284810483f1e60afe34d1d27ad91ce2e/src/diffusers/schedulers/scheduling_utils.py#L51) لـ Stable Diffusion XL هو:

```py
from diffusers.schedulers import AysSchedules

sampling_schedule = AysSchedules["StableDiffusionXLTimesteps"]
print(sampling_schedule)
"[999، 845، 730، 587، 443، 310، 193، 116، 53، 13]"
```

يمكنك استخدام جدول المعايرة AYS في خط أنابيب عن طريق تمريره إلى معلمة "timesteps".

```py
pipeline = StableDiffusionXLPipeline.from_pretrained(
"SG161222/RealVisXL_V4.0"،
torch_dtype=torch.float16،
variant="fp16"،
).to("cuda")
pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config، algorithm_type="sde-dpmsolver++")

prompt = "لقطة سينمائية لأرنب صغير لطيف يرتدي سترة ويقوم بالإبهام"
generator = torch.Generator(device="cpu").manual_seed(2487854446)
image = pipeline(
prompt=prompt،
negative_prompt=""،
generator=generator،
timesteps=sampling_schedule،
).images[0]
```

<div class="flex gap-4">
<div>
<img class="rounded-xl" src="https://huggingface.co/datasets/YiYiXu/testing-images/resolve/main/ays.png"/>
<figcaption class="mt-2 text-center text-sm text-gray-500">جدول الخطوات الزمنية AYS في 10 خطوات</figcaption>
</div>
<div>
<img class="rounded-xl" src="https://huggingface.co/datasets/YiYiXu/testing-images/resolve/main/10.png"/>
<figcaption class="mt-2 text-center text-sm text-gray-500">جدول الخطوات الزمنية المتباعدة خطيًا في 10 خطوات</figcaption>
</div>
<div>
<img class="rounded-xl" src="https://huggingface.co/datasets/YiYiXu/testing-images/resolve/main/25.png"/>
<figcaption class="mt-2 text-center text-sm text-gray-500">جدول الخطوات الزمنية المتباعدة خطيًا في 25 خطوة</figcaption>
</div>
</div>

## تباعد الخطوة الزمنية

يمكن للطريقة التي يتم بها اختيار خطوات المعاينة في الجدول أن تؤثر على جودة الصورة المولدة، خاصة فيما يتعلق [بإعادة تحجيم جدول التشويش](#rescale-noise-schedule)، والذي يمكن أن يمكّن النموذج من إنشاء صور أكثر إشراقًا أو ظلامًا. توفر أجهزة الانتشار ثلاث طرق لتباعد الخطوات الزمنية:

- `leading` يخلق خطوات متباعدة بالتساوي
- `linspace` يتضمن الخطوات الأولى والأخيرة ويحدد بالتساوي الخطوات المتوسطة المتبقية
- `trailing` يتضمن فقط الخطوة الأخيرة ويحدد بالتساوي الخطوات المتوسطة المتبقية بدءًا من النهاية

من المستحسن استخدام طريقة التباعد "trailing" لأنها تولد صورًا ذات جودة أعلى وتفاصيل أكثر عندما يكون هناك عدد أقل من خطوات المعاينة. ولكن الفرق في الجودة ليس واضحًا بنفس القدر لقيم خطوات المعاينة القياسية.

```py
import torch
from diffusers import StableDiffusionXLPipeline، DPMSolverMultistepScheduler

pipeline = StableDiffusionXLPipeline.from_pretrained(
"SG161222/RealVisXL_V4.0"،
torch_dtype=torch.float16،
variant="fp16"،
).to("cuda")
pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config، timestep_spacing="trailing")

prompt = "لقطة سينمائية لقطة سوداء صغيرة لطيفة تجلس على قرع في الليل"
generator = torch.Generator(device="cpu").manual_seed(2487854446)
image = pipeline(
prompt=prompt،
negative_prompt=""،
generator=generator،
num_inference_steps=5،
).images[0]
image
```

<div class="flex gap-4">
<div>
<img class="rounded-xl" src="https://huggingface.co/datasets/stevhliu/testing-images/resolve/main/trailing_spacing.png"/>
<figcaption class="mt-2 text-center text-sm text-gray-500">تباعد متخلف بعد 5 خطوات</figcaption>
</div>
<div>
<img class="rounded-xl" src="https://huggingface.co/datasets/stevhliu/testing-images/resolve/main/leading_spacing.png"/>
<figcaption class="mt-2 text-center text-sm text-gray-500">تباعد رائد بعد 5 خطوات</figcaption>
</div>
</div>

## الانحرافات المعيارية

معلمة "sigmas" هي كمية التشويش المضافة في كل خطوة زمنية وفقًا لجدول الخطوات الزمنية. مثل معلمة "timesteps"، يمكنك تخصيص معلمة "sigmas" للتحكم في مقدار التشويش المضافة في كل خطوة. عندما تستخدم قيمة "sigmas" مخصصة، يتم حساب "timesteps" من قيمة "sigmas" المخصصة ويتم تجاهل تكوين الجدولة الافتراضي.

على سبيل المثال، يمكنك تمرير يدويًا [sigmas](https://github.com/huggingface/diffusers/blob/6529ee67ec02fcf58d2fd9242164ea002b351d75/src/diffusers/schedulers/scheduling_utils.py#L55) لشيء مثل جدول AYS المكون من 10 خطوات من قبل إلى خط الأنابيب.

```py
import torch

from diffusers import DiffusionPipeline، EulerDiscreteScheduler

model_id = "stabilityai/stable-diffusion-xl-base-1.0"
pipeline = DiffusionPipeline.from_pretrained(
"stabilityai/stable-diffusion-xl-base-1.0"،
torch_dtype=torch.float16،
variant="fp16"،
).to("cuda")
pipeline.scheduler = EulerDiscreteScheduler.from_config(pipeline.scheduler.config)

sigmas = [14.615، 6.315، 3.771، 2.181، 1.342، 0.862، 0.555، 0.380، 0.234، 0.113، 0.0]
prompt = "القبيعة الأنثروبومورفية التي ترتدي بدلة وتعمل على جهاز كمبيوتر"
generator = torch.Generator(device='cuda').manual_seed(123)
image = pipeline(
prompt=prompt،
num_inference_steps=10،
sigmas=sigmas،
generator=generator
).images[0]
```

عند إلقاء نظرة على معلمة "timesteps" للجدولة، فستجد أنها مطابقة لجدول الخطوات الزمنية AYS لأن جدول "timestep" يتم حسابه من "sigmas".

```py
print(f" timesteps: {pipe.scheduler.timesteps}")
"timesteps: tensor([999.، 845.، 730.، 587.، 443.، 310.، 193.، 116.، 53.، 13.،]، device='cuda:0')"
```

### الانحرافات المعيارية كاراس

> [!TIP]
> راجع نظرة عامة على [API scheduler](../api/schedulers/overview) للحصول على قائمة بالجداول التي تدعم الانحرافات المعيارية كاراس.
>
> لا ينبغي استخدام الانحرافات المعيارية كاراس للنماذج التي لم يتم تدريبها عليها. على سبيل المثال، لا يجب أن يستخدم نموذج Stable Diffusion XL الأساسي الانحرافات المعيارية كاراس ولكن يمكن لنموذج [DreamShaperXL](https://hf.co/Lykon/dreamshaper-xl-1-0) استخدامها لأنها مدربة على الانحرافات المعيارية كاراس.

تستخدم جداول Karras جدول الخطوات الزمنية والانحرافات المعيارية من الورقة [توضيح مساحة تصميم النماذج المولدة للانتشار](https://hf.co/papers/2206.00364). تطبق هذه المتغيرات من الجدولة كمية أقل من التشويش في كل خطوة حيث تقترب من نهاية عملية المعايرة مقارنة بالجداول الأخرى، ويمكنها زيادة مستوى التفاصيل في الصورة المولدة.

قم بتمكين الانحرافات المعيارية كاراس عن طريق تعيين "use_karras_sigmas=True" في الجدولة.

```py
import torch
from diffusers import StableDiffusionXLPipeline، DPMSolverMultistepScheduler

pipeline = StableDiffusionXLPipeline.from_pretrained(
"SG161222/RealVisXL_V4.0"،
torch_dtype=torch.float16،
variant="fp16"،
).to("cuda")
pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config، algorithm_type="sde-dpmsolver++"، use_karras_sigmas=True)

prompt = "لقطة سينمائية لأرنب صغير لطيف يرتدي سترة ويقوم بالإبهام"
generator = torch.Generator(device="cpu").manual_seed(2487854446)
image = pipeline(
prompt=prompt،
negative_prompt=""،
generator=generator،
).images[0]
```

<div class="flex gap-4">
<div>
<img class="rounded-xl" src="https://huggingface.co/datasets/stevhliu/testing-images/resolve/main/karras_sigmas_true.png"/>
<figcaption class="mt-2 text-center text-sm text-gray-500">تمكين الانحرافات المعيارية كاراس</figcaption>
</div>
<div>
<img class="rounded-xl" src="https://huggingface.co/datasets/stevhliu/testing-images/resolve/main/karras_sigmas_false.png"/>
<figcaption class="mt-2 text-center text-sm text-gray-500">تعطيل الانحرافات المعيارية كاراس</figcaption>
</div>
</div>
بالتأكيد، سأقوم بترجمة النص الموجود في الفقرات والعناوين فقط مع اتباع التعليمات التي قدمتها:

## إعادة ضبط جدول الضجيج

اكتشف المؤلفون في الورقة البحثية [Common Diffusion Noise Schedules and Sample Steps are Flawed](https://hf.co/papers/2305.08891) أن جداول الضجيج الشائعة تسمح بتسرب بعض الإشارة إلى خطوة الوقت الأخيرة. يمكن أن يتسبب هذا التسرب للإشارة أثناء الاستنتاج في جعل النماذج تولد صورًا ذات سطوع متوسط فقط. من خلال فرض نسبة إشارة إلى ضجيج (SNR) تساوي صفرًا لجدول خطوات الوقت وأخذ العينات من خطوة الوقت الأخيرة، يمكن تحسين النموذج لتوليد صور شديدة السطوع أو الظلام.

> [!TIP]
> بالنسبة للاستنتاج، تحتاج إلى نموذج تم تدريبه باستخدام *v_prediction*. لتدريب نموذجك الخاص باستخدام *v_prediction*، أضف علمًا إلى النص التالي في النص البرمجي [train_text_to_image.py](https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image.py) أو [train_text_to_image_lora.py](https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image_lora.py).
>
> ```bash
> --prediction_type="v_prediction"
> ```
> 
على سبيل المثال، قم بتحميل نقطة التفتيش [ptx0/pseudo-journey-v2](https://hf.co/ptx0/pseudo-journey-v2) التي تم تدريبها باستخدام `v_prediction` و [`DDIMScheduler`]. قم بضبط المعلمات التالية في [`DDIMScheduler`]:
* `rescale_betas_zero_snr=True` لإعادة ضبط جدول الضجيج إلى SNR تساوي صفرًا
* `timestep_spacing="trailing"` لبدء أخذ العينات من خطوة الوقت الأخيرة

اضبط `guidance_rescale` في الأنبوب لمنع التعرض الزائد. تقلل القيمة المنخفضة السطوع، ولكن قد تبدو بعض التفاصيل باهتة.

```py
from diffusers import DiffusionPipeline, DDIMScheduler

pipeline = DiffusionPipeline.from_pretrained("ptx0/pseudo-journey-v2", use_safetensors=True)

pipeline.scheduler = DDIMScheduler.from_config(
pipeline.scheduler.config, rescale_betas_zero_snr=True, timestep_spacing="trailing"
)
pipeline.to("cuda")
prompt = "cinematic photo of a snowy mountain at night with the northern lights aurora borealis overhead, 35mm photograph, film, professional, 4k, highly detailed"
generator = torch.Generator(device="cpu").manual_seed(23)
image = pipeline(prompt, guidance_rescale=0.7, generator=generator).images[0]
image
```

<div class="flex gap-4">
<div>
<img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/no-zero-snr.png"/>
<figcaption class="mt-2 text-center text-sm text-gray-500">صورة Stable Diffusion v2-1 الافتراضية</figcaption>
</div>
<div>
<img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/zero-snr.png"/>
<figcaption class="mt-2 text-center text-sm text-gray-500">صورة مع SNR تساوي صفرًا و trailing timestep spacing مفعلة</figcaption>
</div>
</div>