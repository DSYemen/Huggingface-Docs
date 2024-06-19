لم يتم ترجمة الأجزاء المحددة في الطلب:

# تقليل استخدام الذاكرة

تمثل الكمية الكبيرة من الذاكرة المطلوبة عائقا أمام استخدام نماذج الانتشار. وللتغلب على هذا التحدي، هناك العديد من التقنيات التي يمكنك استخدامها لتقليل الذاكرة لتشغيل حتى أكبر النماذج على وحدات معالجة الرسوميات (GPU) من الفئة المجانية أو الاستهلاكية. ويمكن حتى الجمع بين بعض هذه التقنيات لزيادة تقليل استخدام الذاكرة.

في العديد من الحالات، يؤدي التحسين من أجل الذاكرة أو السرعة إلى تحسين الأداء في الجانب الآخر، لذلك يجب عليك محاولة التحسين للاثنين معًا كلما استطعت. ويركز هذا الدليل على تقليل استخدام الذاكرة إلى الحد الأدنى، ولكن يمكنك أيضًا معرفة المزيد حول كيفية [تسريع الاستنتاج](fp16).

تم الحصول على النتائج أدناه من خلال إنشاء صورة واحدة بحجم 512x512 من موجه "صورة رائد فضاء يركب حصانًا على المريخ" مع 50 خطوة DDIM على Nvidia Titan RTX، مما يوضح التسريع الذي يمكن توقعه نتيجة لانخفاض استهلاك الذاكرة.

|                  | الكمون | التسريع |
| ---------------- | ------- | ------- |
| الأصلي         | 9.50 ثانية   | 1 مرة      |
| fp16             | 3.61 ثانية   | 2.63 مرة   |
| القنوات الأخيرة    | 3.30 ثانية   | 2.88 مرة   |
| تتبع UNet      | 3.21 ثانية   | 2.96 مرة   |
| انتباه فعال من حيث التكلفة للذاكرة | 2.63 ثانية  | 3.61 مرات   |

## VAE المقطوع

تمكّن VAE المقطوعة من فك تشفير دفعات كبيرة من الصور بذاكرة VRAM محدودة أو دفعات تحتوي على 32 صورة أو أكثر عن طريق فك تشفير دفعات المخفونات صورة واحدة في كل مرة. من المحتمل أن ترغب في اقتران هذا بـ [`~ModelMixin.enable_xformers_memory_efficient_attention`] لتقليل استخدام الذاكرة بشكل أكبر إذا كان لديك xFormers مثبتًا.

لاستخدام VAE المقطوع، اتصل بـ [`~StableDiffusionPipeline.enable_vae_slicing`] على خط أنابيبك قبل الاستدلال:

```python
import torch
from diffusers import StableDiffusionPipeline

pipeline = StableDiffusionPipeline.from_pretrained(
"runwayml/stable-diffusion-v1-5",
torch_dtype=torch.float16,
use_safetensors=True,
)
pipe = pipeline.to("cuda")

prompt = "صورة لرائد فضاء يركب حصانًا على المريخ"
pipe.enable_vae_slicing()
#pipe.enable_xformers_memory_efficient_attention()
images = pipe([prompt] * 32).images
```

قد تشاهد زيادة طفيفة في الأداء في فك تشفير VAE على دفعات متعددة الصور، ولا يجب أن يكون هناك أي تأثير على الأداء في دفعات الصور الفردية.

## VAE المبلط

تمكين معالجة VAE المبلطة أيضًا العمل مع الصور الكبيرة بذاكرة VRAM محدودة (على سبيل المثال، إنشاء صور 4K بذاكرة وصول عشوائي (VRAM) سعة 8 جيجابايت) عن طريق تقسيم الصورة إلى بلاطات متداخلة، وفك تشفير البلاطات، ثم مزج المخرجات معًا لتكوين الصورة النهائية. يجب أيضًا استخدام VAE المبلط مع [`~ModelMixin.enable_xformers_memory_efficient_attention`] لتقليل استخدام الذاكرة بشكل أكبر إذا كان لديك xFormers مثبتًا.

لاستخدام معالجة VAE المبلطة، اتصل بـ [`~StableDiffusionPipeline.enable_vae_tiling`] على خط أنابيبك قبل الاستدلال:

```python
import torch
from diffusers import StableDiffusionPipeline, UniPCMultistepScheduler

pipe = StableDiffusionPipeline.from_pretrained(
"runwayml/stable-diffusion-v1-5",
torch_dtype=torch.float16,
use_safetensors=True,
)
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to("cuda")
prompt = "صورة جميلة للمناظر الطبيعية"
pipe.enable_vae_tiling()
#pipe.enable_xformers_memory_efficient_attention()

image = pipe([prompt], width=3840, height=2224, num_inference_steps=20)images [0]
```

تحتوي صورة الإخراج على بعض الاختلافات في نغمة البلاط نظرًا لفك تشفير البلاطات بشكل منفصل، ولكن لا يجب أن تشاهد أي درزات حادة وواضحة بين البلاطات. يتم إيقاف التبليط للصور التي يبلغ حجمها 512x512 أو أصغر.

## التفريغ على وحدة المعالجة المركزية

يمكن أيضًا توفير الذاكرة عن طريق تفريغ الأوزان إلى وحدة المعالجة المركزية (CPU) وتحميلها فقط على وحدة معالجة الرسوميات (GPU) عند إجراء تمرير للأمام. وغالبًا ما يمكن أن تقلل هذه التقنية استهلاك الذاكرة إلى أقل من 3 جيجابايت.

لأداء التفريغ على وحدة المعالجة المركزية، اتصل بـ [`~StableDiffusionPipeline.enable_sequential_cpu_offload`]:

```Python
import torch
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained(
"runwayml/stable-diffusion-v1-5",
torch_dtype=torch.float16,
use_safetensors=True,
)

prompt = "صورة لرائد فضاء يركب حصانًا على المريخ"
pipe.enable_sequential_cpu_offload()
image = pipe(prompt).images [0]
```

يعمل التفريغ على وحدة المعالجة المركزية على الوحدات الفرعية بدلاً من النماذج الكاملة. هذه هي الطريقة المثلى لتقليل استهلاك الذاكرة، ولكن الاستدلال أبطأ بكثير بسبب الطبيعة التكرارية لعملية الانتشار. تقوم مكون Unet من خط الأنابيب بتشغيل عدة مرات (مثلما هو الحال `num_inference_steps`)؛ في كل مرة، يتم تحميل الوحدات الفرعية Unet وتحميلها بشكل متسلسل حسب الحاجة، مما يؤدي إلى عدد كبير من عمليات نقل الذاكرة.

فكر في استخدام [تفريغ النموذج](#model-offloading) إذا كنت تريد التحسين للسرعة لأنه أسرع بكثير. المقايضة هي أن وفورات الذاكرة الخاصة بك لن تكون كبيرة.

<Tip warning={true}>

عند استخدام [`~StableDiffusionPipeline.enable_sequential_cpu_offload`]، لا تنقل خط الأنابيب إلى CUDA مسبقًا وإلا ستكون الزيادة في استهلاك الذاكرة طفيفة فقط (راجع هذا [القضية](https://github.com/huggingface/diffusers/issues/1934) لمزيد من المعلومات).

[`~StableDiffusionPipeline.enable_sequential_cpu_offload`] هي عملية ذات حالة تقوم بتثبيت الخطافات على النماذج.

</Tip>

## تفريغ النموذج

<Tip>

يتطلب تفريغ النموذج 🤗 Accelerate الإصدار 0.17.0 أو أعلى.

</Tip>

يحافظ [التفريغ التسلسلي لوحدة المعالجة المركزية](#cpu-offloading) على الكثير من الذاكرة ولكنه يجعل الاستدلال أبطأ لأن الوحدات الفرعية يتم نقلها إلى وحدة معالجة الرسوميات (GPU) حسب الحاجة، ويتم إرجاعها على الفور إلى وحدة المعالجة المركزية (CPU) عندما يقوم وحدة معالجة الرسوميات (GPU) بتشغيل وحدة جديدة.

والتفريغ الكامل للنموذج هو بديل ينقل النماذج الكاملة إلى وحدة معالجة الرسوميات (GPU)، بدلاً من التعامل مع الوحدات الفرعية لكل نموذج. يوجد تأثير ضئيل على وقت الاستدلال (مقارنة بنقل خط الأنابيب إلى `cuda`)، ولا يزال يوفر بعض وفورات الذاكرة.

أثناء تفريغ النموذج، يتم وضع أحد المكونات الرئيسية لخط الأنابيب فقط (عادةً ما يكون الترميز النصي، وUnet وVAE)
يتم وضعها على وحدة معالجة الرسوميات (GPU) في حين تنتظر المكونات الأخرى على وحدة المعالجة المركزية (CPU). تظل المكونات مثل Unet، التي تعمل لعدة تكرارات، على وحدة معالجة الرسوميات (GPU) حتى لا تكون هناك حاجة إليها.

قم بتمكين تفريغ النموذج عن طريق استدعاء [`~StableDiffusionPipeline.enable_model_cpu_offload`] على خط الأنابيب:

```Python
import torch
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained(
"runwayml/stable-diffusion-v1-5",
torch_dtype=torch.float16,
use_safetensors=True,
)

prompt = "صورة لرائد فضاء يركب حصانًا على المريخ"
pipe.enable_model_cpu_offload()
image = pipe(prompt).images [0]
```

<Tip warning={true}>

لإلغاء تحميل النماذج بشكل صحيح بعد استدعائها، من الضروري تشغيل خط الأنابيب بأكمله، ويتم استدعاء النماذج في الترتيب المتوقع لخط الأنابيب. توخ الحذر إذا تمت إعادة استخدام النماذج خارج سياق خط الأنابيب بعد تثبيت الخطافات. راجع [إزالة الخطافات](https://huggingface.co/docs/accelerate/en/package_reference/big_modeling#accelerate.hooks.remove_hook_from_module) لمزيد من المعلومات.

[`~StableDiffusionPipeline.enable_model_cpu_offload`] هي عملية ذات حالة تقوم بتثبيت الخطافات على النماذج والحالة على خط الأنابيب.

</Tip>

## تنسيق الذاكرة للقنوات الأخيرة

تنسيق الذاكرة للقنوات الأخيرة هو طريقة بديلة لترتيب تنسيقات NCHW في الذاكرة للحفاظ على ترتيب الأبعاد. يتم ترتيب القنوات الأخيرة بحيث تصبح القنوات البعد الأكثر كثافة (تخزين الصور بكسل لكل بكسل). نظرًا لأن المشغلين لا يدعمون حاليًا تنسيق القنوات الأخيرة، فقد يؤدي ذلك إلى أسوأ أداء، ولكن يجب عليك مع ذلك تجربته لمعرفة ما إذا كان يعمل لنموذجك.

على سبيل المثال، لتعيين تنسيق القنوات الأخيرة لشبكة Unet في خط الأنابيب:

```python
print(pipe.unet.conv_out.state_dict () ["weight"].step()) # (2880، 9، 3، 1)
pipe.unet.to(memory_format=torch.channels_last) # عملية في المكان
print(pipe.unet.conv_out.state_dict()["weight"].step()) # (2880، 1، 960، 320) وجود خطوة 1 للبعد 2 يثبت أنه يعمل
```
## تتبع

يقوم التتبع بتشغيل مثال على إدخال tensor عبر النموذج ويحصل على العمليات التي يتم تنفيذها عليه أثناء مروره عبر طبقات النموذج. ويتم تحسين القابل للتنفيذ أو `ScriptFunction` الذي يتم إرجاعه باستخدام التجميع في الوقت المناسب.

لتتبع UNet:

... [Code Python] ...

## الانتباه الفعال من حيث الذاكرة

لقد حققت الأعمال الحديثة في تحسين عرض النطاق الترددي في كتلة الاهتمام تسريعًا كبيرًا وتخفيضات في استخدام ذاكرة GPU. وأحدث نوع من الاهتمام الفعال من حيث الذاكرة هو [Flash Attention] (https://arxiv.org/abs/2205.14135) (يمكنك الاطلاع على الكود الأصلي في [HazyResearch/flash-attention] (https://github.com/HazyResearch/flash-attention)).

<Tip>

إذا كان لديك PyTorch >= 2.0 مثبتًا، فلا ينبغي أن تتوقع تسريعًا للاستدلال عند تمكين `xformers`.

</Tip>

لاستخدام Flash Attention، قم بتثبيت ما يلي:

- Pytorch > 1.12
- CUDA متاح
- [xFormers] (xformers)

ثم استدعاء [`~ ModelMixin.enable_xformers_memory_efficient_attention`] على الأنابيب:

... [Code Python] ...

يجب أن تتطابق سرعة التكرار عند استخدام `xformers` مع سرعة تكرار PyTorch 2.0 كما هو موضح [هنا] (torch2.0).