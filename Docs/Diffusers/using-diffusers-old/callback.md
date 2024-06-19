# استدعاءات خط الأنابيب

يمكن تعديل حلقة إزالة التشويش في خط الأنابيب باستخدام الدالات المحددة مخصصًا باستخدام معامل `callback_on_step_end`. يتم تنفيذ دالة الاستدعاء في نهاية كل خطوة، وتعديل سمات خط الأنابيب ومتغيرات للخطوة التالية. هذا مفيد حقًا لتعديل بعض سمات خط الأنابيب أو متغيرات المنسوجة ديناميكيًا. تسمح هذه المرونة بحالات استخدام مثيرة للاهتمام مثل تغيير تضمينات المطالبة في كل خطوة زمنية، وتعيين أوزان مختلفة لتضمينات المطالبة، وتحرير مقياس التوجيه. باستخدام الاستدعاءات، يمكنك تنفيذ ميزات جديدة دون تعديل التعليمات البرمجية الأساسية!

> [!TIP]
> في الوقت الحالي، تدعم تطبيقات 🤗 فقط `callback_on_step_end`، ولكن لا تتردد في فتح [طلب ميزة](https://github.com/huggingface/diffusers/issues/new/choose) إذا كان لديك حالة استخدام رائعة وتتطلب دالة استدعاء بنقطة تنفيذ مختلفة!

سيوضح هذا الدليل كيفية عمل الاستدعاءات من خلال بعض الميزات التي يمكنك تنفيذها باستخدامها.

## الاستدعاءات الرسمية

نقدم قائمة من الاستدعاءات التي يمكنك توصيلها بخط أنابيب موجود وتعديل حلقة إزالة التشويش. هذه هي القائمة الحالية من الاستدعاءات الرسمية:

- `SDCFGCutoffCallback`: يعطل CFG بعد عدد معين من الخطوات لجميع خطوط أنابيب SD 1.5، بما في ذلك النص إلى الصورة، والصورة إلى الصورة، والإكمال، وControlNet.
- `SDXLCFGCutoffCallback`: يعطل CFG بعد عدد معين من الخطوات لجميع خطوط أنابيب SDXL، بما في ذلك النص إلى الصورة، والصورة إلى الصورة، والإكمال، وControlNet.
- `IPAdapterScaleCutoffCallback`: يعطل محول IP بعد عدد معين من الخطوات لجميع خطوط الأنابيب التي تدعم محول IP.

> [!TIP]
> إذا كنت تريد إضافة استدعاء رسمي جديد، فلا تتردد في فتح [طلب ميزة](https://github.com/huggingface/diffusers/issues/new/choose) أو [تقديم طلب سحب](https://huggingface.co/docs/diffusers/main/en/conceptual/contribution#how-to-open-a-pr).

لإعداد استدعاء، تحتاج إلى تحديد عدد خطوات إزالة التشويش التي يصبح فيها الاستدعاء ساري المفعول. يمكنك القيام بذلك باستخدام أي من حججي التاليين:

- `cutoff_step_ratio`: رقم عائم مع نسبة الخطوات.
- `cutoff_step_index`: رقم صحيح مع رقم الخطوة الدقيق.

```python
import torch

from diffusers import DPMSolverMultistepScheduler, StableDiffusionXLPipeline
from diffusers.callbacks import SDXLCFGCutoffCallback


callback = SDXLCFGCutoffCallback(cutoff_step_ratio=0.4)
# يمكن أيضًا استخدامه مع cutoff_step_index
# الاستدعاء = SDXLCFGCutoffCallback(cutoff_step_ratio=None، cutoff_step_index=10)

pipeline = StableDiffusionXLPipeline.from_pretrained(
"stabilityai/stable-diffusion-xl-base-1.0"،
torch_dtype=torch.float16،
variant="fp16"،
).to("cuda")
pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config، use_karras_sigmas=True)

المطالبة = "سيارة رياضية على الطريق، أفضل جودة، جودة عالية، تفاصيل عالية، دقة 8k"

المولد = torch.Generator(device="cpu").manual_seed(2628670641)

الناتج = pipeline(
المطالبة = المطالبة،
negative_prompt=""،
guidance_scale=6.5،
num_inference_steps=25،
generator=generator،
callback_on_step_end=callback،
)

out.images[0].save("official_callback.png")
```

<div class="flex gap-4">
<div>
<img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/without_cfg_callback.png" alt="الصورة المولدة لسيارة رياضية على الطريق" />
<figcaption class="mt-2 text-center text-sm text-gray-500">بدون SDXLCFGCutoffCallback</figcaption>
</div>
<div>
<img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/with_cfg_callback.png" alt="الصورة المولدة لسيارة رياضية على الطريق مع استدعاء CFG" />
<figcaption class="mt-2 text-center text-sm text-gray-500">مع SDXLCFGCutoffCallback</figcaption>
</div>
</div>

## التوجيه الحر الديناميكي للتصنيف

التوجيه الحر الديناميكي للتصنيف (CFG) هو ميزة تسمح لك بتعطيل CFG بعد عدد معين من خطوات الاستدلال والتي يمكن أن تساعدك على توفير الطاقة الحسابية بتكلفة ضئيلة للأداء. يجب أن تحتوي دالة الاستدعاء لهذا على الحجج التالية:

- `pipeline` (أو مثيل خط الأنابيب) يوفر الوصول إلى الخصائص المهمة مثل `num_timesteps` و`guidance_scale`. يمكنك تعديل هذه الخصائص عن طريق تحديث السمات الأساسية. بالنسبة لهذا المثال، ستعطل CFG عن طريق تعيين `pipeline._guidance_scale=0.0`.
- `step_index` و`timestep` يخبرانك أين أنت في حلقة إزالة التشويش. استخدم `step_index` لإيقاف تشغيل CFG بعد الوصول إلى 40% من `num_timesteps`.
- `callback_kwargs` هو قاموس يحتوي على متغيرات المنسوجة التي يمكنك تعديلها أثناء حلقة إزالة التشويش. فهو لا يشمل سوى المتغيرات المحددة في حجة `callback_on_step_end_tensor_inputs`، والتي يتم تمريرها إلى طريقة `__call__` لخط الأنابيب. قد تستخدم خطوط الأنابيب المختلفة مجموعات مختلفة من المتغيرات، لذا يرجى التحقق من سمة `_callback_tensor_inputs` لخط الأنابيب للحصول على قائمة بالمتغيرات التي يمكنك تعديلها. بعض المتغيرات الشائعة تشمل `latents` و`prompt_embeds`. بالنسبة لهذه الدالة، قم بتغيير حجم دفعة `prompt_embeds` بعد تعيين `guidance_scale=0.0` لكي تعمل بشكل صحيح.

يجب أن تبدو دالة الاستدعاء الخاصة بك شيئًا مثل هذا:

```python
def callback_dynamic_cfg(pipe، step_index، timestep، callback_kwargs):
# ضبط حجم دفعة prompt_embeds وفقًا لـ guidance_scale
if step_index == int(pipeline.num_timesteps * 0.4):
prompt_embeds = callback_kwargs["prompt_embeds"]
prompt_embeds = prompt_embeds.chunk(2)[-1]

# تحديث guidance_scale وprompt_embeds
pipeline._guidance_scale = 0.0
callback_kwargs["prompt_embeds"] = prompt_embeds
return callback_kwargs
```

الآن، يمكنك تمرير دالة الاستدعاء إلى معلمة `callback_on_step_end` و`prompt_embeds` إلى `callback_on_step_end_tensor_inputs`.

```بي
استيراد الشعلة
من الناشرين import StableDiffusionPipeline

pipeline = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5"، torch_dtype=torch.float16)
pipeline = pipeline.to("cuda")

المطالبة = "صورة لرائد فضاء يركب حصانًا على المريخ"

المولد = torch.Generator(device="cuda").manual_seed(1)
الناتج = pipeline(
المطالبة،
generator=generator،
callback_on_step_end=callback_dynamic_cfg،
callback_on_step_end_tensor_inputs=['prompt_embeds']
)

out.images[0].save("out_custom_cfg.png")
```

## مقاطعة عملية الانتشار

> [!TIP]
> يتم دعم استدعاء مقاطعة لترجمة النص إلى صورة، والصورة إلى الصورة، والإكمال لـ [StableDiffusionPipeline](../api/pipelines/stable_diffusion/overview) و[StableDiffusionXLPipeline](../api/pipelines/stable_diffusion/stable_diffusion_xl).

إن إيقاف عملية الانتشار مبكرًا مفيد عند بناء واجهات المستخدم التي تعمل مع تطبيقات 🤗 لأنها تسمح للمستخدمين بإيقاف عملية التوليد إذا لم يكونوا راضين عن النتائج المتوسطة. يمكنك دمج هذا في خط أنابيبك باستخدام استدعاء.

يجب أن تأخذ دالة الاستدعاء هذه الحجج: `pipeline`، `i`، `t`، و`callback_kwargs` (يجب إرجاعها). قم بتعيين سمة `_interrupt` لخط الأنابيب إلى `True` لوقف عملية الانتشار بعد عدد معين من الخطوات. أنت حر أيضًا في تنفيذ منطق التوقف المخصص الخاص بك داخل الاستدعاء.

في هذا المثال، يتم إيقاف عملية الانتشار بعد 10 خطوات على الرغم من تعيين `num_inference_steps` إلى 50.

```python
from diffusers import StableDiffusionPipeline

pipeline = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
pipeline.enable_model_cpu_offload()
num_inference_steps = 50

def interrupt_callback(pipeline، i، t، callback_kwargs):
stop_idx = 10
if i == stop_idx:
pipeline._interrupt = True

return callback_kwargs

pipeline(
"صورة قطة"،
num_inference_steps=num_inference_steps،
callback_on_step_end=interrupt_callback،
)
```
## عرض الصورة بعد كل خطوة من خطوات التوليد

قم بعرض صورة بعد كل خطوة من خطوات التوليد عن طريق الوصول إلى القيم الكامنة وتحويلها إلى صورة بعد كل خطوة. يتم ضغط مساحة القيم الكامنة إلى 128x128، لذلك تكون الصور أيضًا بحجم 128x128، وهو ما يفيد في المعاينة السريعة.

1. استخدم الدالة أدناه لتحويل القيم الكامنة SDXL (4 قنوات) إلى مصفوفات RGB (3 قنوات) كما هو موضح في منشور المدونة [Explaining the SDXL latent space](https://huggingface.co/blog/TimothyAlexisVass/explaining-the-sdxl-latent-space).

   ```py
   غير مترجم
   ```

2. قم بإنشاء دالة لترميز القيم الكامنة وحفظها في صورة.

   ```py
   غير مترجم
   ```

3. قم بتمرير دالة `decode_tensors` إلى معلمة `callback_on_step_end` لترميز المصفوفات بعد كل خطوة. كما تحتاج أيضًا إلى تحديد ما تريد تعديله في معلمة `callback_on_step_end_tensor_inputs`، والتي في هذه الحالة هي القيم الكامنة.

   ```py
   غير مترجم
   ```

<div class="flex gap-4 justify-center">
 <div>
  <img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/tips_step_0.png"/>
  <figcaption class="mt-2 text-center text-sm text-gray-500">الخطوة 0</figcaption>
 </div>
 <div>
  <img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/tips_step_19.png"/>
  <figcaption class="mt-2 text-center text-sm text-gray-500">الخطوة 19</figcaption>
 </div>
 <div>
  <img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/tips_step_29.png"/>
  <figcaption class="mt-2 text-center text-sm text-gray-500">الخطوة 29</figcaption>
 </div>
 <div>
  <img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/tips_step_39.png"/>
  <figcaption class="mt-2 text-center text-sm text-gray-500">الخطوة 39</figcaption>
 </div>
 <div>
  <img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/tips_step_49.png"/>
  <figcaption class="mt-2 text-center text-sm text-gray-500">الخطوة 49</figcaption>
 </div>
</div>