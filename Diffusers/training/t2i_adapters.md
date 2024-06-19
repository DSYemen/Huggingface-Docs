# T2I-Adapter 

T2I-Adapter هو نموذج خفيف الوزن يوفر إدخال صورة شرطية إضافية (رسوم الخطوط، أو Canny، أو Sketch، أو Depth، أو Pose) للتحكم بشكل أفضل في توليد الصور. وهو يشبه ControlNet، ولكنه أصغر بكثير (حوالي 77 مليون معامل وحجم ملف يبلغ حوالي 300 ميجابايت) لأنه يقوم بإدراج الأوزان فقط في شبكة U-Net بدلاً من نسخها وتدريبها.

يتوفر T2I-Adapter للتدريب فقط مع نموذج Stable Diffusion XL (SDXL).

سيتعمق هذا الدليل في دراسة نص البرنامج النصي للتدريب [train_t2i_adapter_sdxl.py](https://github.com/huggingface/diffusers/blob/main/examples/t2i_adapter/train_t2i_adapter_sdxl.py) لمساعدتك على التعرف عليه، وكيف يمكنك تكييفه مع حالتك الاستخدام الخاصة.

قبل تشغيل البرنامج النصي، تأكد من تثبيت المكتبة من المصدر:

انتقل إلى مجلد المثال الذي يحتوي على البرنامج النصي للتدريب وقم بتثبيت التبعيات المطلوبة للبرنامج النصي الذي تستخدمه:

<Tip>
🤗 Accelerate هي مكتبة تساعدك على التدريب على وحدات GPU/TPUs متعددة أو مع الدقة المختلطة. سيقوم تلقائيًا بتكوين إعداد التدريب الخاص بك بناءً على أجهزتك وبيئتك. الق نظرة على جولة 🤗 Accelerate السريعة [Quick tour](https://huggingface.co/docs/accelerate/quicktour) لمعرفة المزيد.
</Tip>

قم بتهيئة بيئة 🤗 Accelerate:

أو قم بتهيئة بيئة 🤗 Accelerate الافتراضية دون اختيار أي تكوينات:

أو إذا لم يدعم بيئتك غلافًا تفاعليًا، مثل دفتر الملاحظات، فيمكنك استخدام ما يلي:

أخيرًا، إذا كنت تريد تدريب نموذج على مجموعة البيانات الخاصة بك، فراجع دليل [إنشاء مجموعة بيانات للتدريب](create_dataset) لمعرفة كيفية إنشاء مجموعة بيانات تعمل مع البرنامج النصي للتدريب.

<Tip>
تسلط الأقسام التالية الضوء على أجزاء من البرنامج النصي للتدريب والتي تعد مهمة لفهم كيفية تعديلها، ولكنها لا تغطي كل جانب من جوانب البرنامج النصي بالتفصيل. إذا كنت مهتمًا بمعرفة المزيد، فلا تتردد في قراءة البرنامج النصي [script](https://github.com/huggingface/diffusers/blob/main/examples/t2i_adapter/train_t2i_adapter_sdxl.py) وأخبرنا إذا كان لديك أي أسئلة أو مخاوف.
</Tip>

## معلمات البرنامج النصي

يوفر البرنامج النصي للتدريب العديد من المعلمات لمساعدتك على تخصيص عملية تشغيل التدريب. يمكن العثور على جميع المعلمات ووصفها في دالة [`parse_args()`](https://github.com/huggingface/diffusers/blob/aab6de22c33cc01fb7bc81c0807d6109e2c998c9/examples/t2i_adapter/train_t2i_adapter_sdxl.py#L233). يوفر قيم افتراضية لكل معلمة، مثل حجم دفعة التدريب ومعدل التعلم، ولكن يمكنك أيضًا تعيين قيمك الخاصة في أمر التدريب إذا أردت.

على سبيل المثال، لتنشيط تجميع التدرجات، أضف المعلمة `--gradient_accumulation_steps` إلى أمر التدريب:

يتم وصف العديد من المعلمات الأساسية والمهمة في دليل التدريب [Text-to-image](text2image#script-parameters)، لذلك يركز هذا الدليل فقط على معلمات T2I-Adapter ذات الصلة:

- `--pretrained_vae_model_name_or_path`: المسار إلى VAE مُدرب مسبقًا؛ من المعروف أن VAE الخاص بـ SDXL يعاني من عدم استقرار رقمي، لذلك تسمح هذه المعلمة بتحديد VAE أفضل [VAE](https://huggingface.co/madebyollin/sdxl-vae-fp16-fix)

- `--crops_coords_top_left_h` و `--crops_coords_top_left_w`: إحداثيات الارتفاع والعرض المراد تضمينها في تضمين إحداثيات القطع SDXL

- `--conditioning_image_column`: عمود الصور الشرطية في مجموعة البيانات

- `--proportion_empty_prompts`: نسبة موجهات الصور ليتم استبدالها بسلسلة فارغة

## البرنامج النصي للتدريب

كما هو الحال مع معلمات البرنامج النصي، يتم توفير دليل تفصيلي للبرنامج النصي للتدريب في دليل التدريب [Text-to-image](text2image#training-script). بدلاً من ذلك، يلقي هذا الدليل نظرة على أجزاء البرنامج النصي ذات الصلة بـ T2I-Adapter.

يبدأ البرنامج النصي للتدريب عن طريق إعداد مجموعة البيانات. ويشمل ذلك [معالجة](https://github.com/huggingface/diffusers/blob/aab6de22c33cc01fb7bc81c0807d6109e2c998c9/examples/t2i_adapter/train_t2i_adapter_sdxl.py#L674) النص وإجراء [التحويلات](https://github.com/huggingface/diffusers/blob/aab6de22c33cc01fb7bc81c0807d6109e2c998c9/examples/t2i_adapter/train_t2i_adapter_sdxl.py#L714) على الصور والصور الشرطية.

ضمن دالة [`main()`](https://github.com/huggingface/diffusers/blob/aab6de22c33cc01fb7bc81c0807d6109e2c998c9/examples/t2i_adapter/train_t2i_adapter_sdxl.py#L770)، يتم تحميل T2I-Adapter إما من محول مُدرب مسبقًا أو يتم تهيئته بشكل عشوائي:

يتم تهيئة [المحسن](https://github.com/huggingface/diffusers/blob/aab6de22c33cc01fb7bc81c0807d6109e2c998c9/examples/t2i_adapter/train_t2i_adapter_sdxl.py#L952) لمعلمات T2I-Adapter:

أخيرًا، في [حلقة التدريب](https://github.com/huggingface/diffusers/blob/aab6de22c33cc01fb7bc81c0807d6109e2c998c9/examples/t2i_adapter/train_t2i_adapter_sdxl.py#L1086)، يتم تمرير الصورة الشرطية للنص ومحولات النص إلى شبكة U-Net للتنبؤ ببقايا الضوضاء:

إذا كنت تريد معرفة المزيد حول كيفية عمل حلقة التدريب، فراجع البرنامج التعليمي [Understanding pipelines, models and schedulers](../using-diffusers/write_own_pipeline) الذي يقوم بتفسيك النمط الأساسي لعملية إزالة الضوضاء.
## تشغيل البرنامج النصي

الآن أنت مستعد لتشغيل برنامج التدريب النصي! 🚀

بالنسبة لهذا التدريب التوضيحي، سوف تستخدم مجموعة البيانات [fusing/fill50k](https://huggingface.co/datasets/fusing/fill50k). يمكنك أيضًا إنشاء مجموعة البيانات الخاصة بك واستخدامها إذا أردت (راجع دليل [إنشاء مجموعة بيانات للتدريب](https://moon-ci-docs.huggingface.co/docs/diffusers/pr_5512/en/training/create_dataset)).

قم بتعيين متغير البيئة `MODEL_DIR` إلى معرف نموذج على Hub أو مسار إلى نموذج محلي و`OUTPUT_DIR` إلى المكان الذي تريد حفظ النموذج فيه.

قم بتنزيل الصور التالية لتهيئة التدريب الخاص بك:

```bash
wget https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/controlnet_training/conditioning_image_1.png
wget https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/controlnet_training/conditioning_image_2.png
```

<Tip>

لمراقبة تقدم التدريب باستخدام Weights & Biases، أضف المعلمة `--report_to=wandb` إلى أمر التدريب. ستحتاج أيضًا إلى إضافة `--validation_image`، و`--validation_prompt`، و`--validation_steps` إلى أمر التدريب لتتبع النتائج. يمكن أن يكون هذا مفيدًا جدًا في تصحيح أخطاء النموذج وعرض النتائج الوسيطة.

</Tip>

```bash
export MODEL_DIR="stabilityai/stable-diffusion-xl-base-1.0"
export OUTPUT_DIR="path to save model"

accelerate launch train_t2i_adapter_sdxl.py \
--pretrained_model_name_or_path=$MODEL_DIR \
--output_dir=$OUTPUT_DIR \
--dataset_name=fusing/fill50k \
--mixed_precision="fp16" \
--resolution=1024 \
--learning_rate=1e-5 \
--max_train_steps=15000 \
--validation_image "./conditioning_image_1.png" "./conditioning_image_2.png" \
--validation_prompt "red circle with blue background" "cyan circle with brown floral background" \
--validation_steps=100 \
--train_batch_size=1 \
--gradient_accumulation_steps=4 \
--report_to="wandb" \
--seed=42 \
--push_to_hub
```

بمجرد اكتمال التدريب، يمكنك استخدام T2I-Adapter للتنبؤ:

```py
from diffusers import StableDiffusionXLAdapterPipeline, T2IAdapter, EulerAncestralDiscreteSchedulerTest
from diffusers.utils import load_image
import torch

adapter = T2IAdapter.from_pretrained("path/to/adapter", torch_dtype=torch.float16)
pipeline = StableDiffusionXLAdapterPipeline.from_pretrained(
"stabilityai/stable-diffusion-xl-base-1.0", adapter=adapter, torch_dtype=torch.float16
)

pipeline.scheduler = EulerAncestralDiscreteSchedulerTest.from_config(pipe.scheduler.config)
pipeline.enable_xformers_memory_efficient_attention()
pipeline.enable_model_cpu_offload()

control_image = load_image("./conditioning_image_1.png")
prompt = "pale golden rod circle with old lace background"

generator = torch.manual_seed(0)
image = pipeline(
prompt, image=control_image, generator=generator
).images[0]
image.save("./output.png")
```

## الخطوات التالية

تهانينا على تدريب نموذج T2I-Adapter! 🎉 لمزيد من المعلومات:

- اقرأ منشور المدونة [Efficient Controllable Generation for SDXL with T2I-Adapters](https://huggingface.co/blog/t2i-sdxl-adapters) لمعرفة المزيد من التفاصيل حول النتائج التجريبية من فريق T2I-Adapter.