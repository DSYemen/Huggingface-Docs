## الاستدلال الموزع باستخدام عدة وحدات معالجة الرسومات (GPU)

في الإعدادات الموزعة، يمكنك تشغيل الاستدلال عبر وحدات معالجة الرسومات (GPU) متعددة باستخدام Accelerate أو PyTorch Distributed، وهو ما يفيد في توليد موجهات متعددة بالتوازي. سيوضح هذا الدليل كيفية استخدام Accelerate وPyTorch Distributed للاستدلال الموزع.

### Accelerate

Accelerate هي مكتبة مصممة لتسهيل التدريب أو تشغيل الاستدلال عبر الإعدادات الموزعة. فهو يبسط عملية إعداد البيئة الموزعة، مما يتيح لك التركيز على تعليمات برمجة PyTorch الخاصة بك.

للبدء، قم بإنشاء ملف Python وقم بتهيئة "accelerate.PartialState" لإنشاء بيئة موزعة؛ يتم اكتشاف إعدادك تلقائيًا، لذلك لا تحتاج إلى تحديد "الرتبة" (rank) أو "حجم العالم" (world_size) بشكل صريح. قم بنقل "DiffusionPipeline" إلى "distributed_state.device" لتعيين وحدة معالجة الرسومات (GPU) لكل عملية.

الآن، استخدم أداة "accelerate.PartialState.split_between_processes" كمدير سياق لتوزيع الموجهات تلقائيًا بين عدد العمليات.

```py
import torch
from accelerate import PartialState
from diffusers import DiffusionPipeline

pipeline = DiffusionPipeline.from_pretrained(
"runwayml/stable-diffusion-v1-5"، torch_dtype=torch.float16، use_safetensors=True
)
distributed_state = PartialState()
pipeline.to(distributed_state.device)

مع distributed_state.split_between_processes (["a dog"، "a cat"]) كموجه:
النتيجة = pipeline (prompt).images [0]
النتيجة. save (f "result_ {distributed_state.process_index}. png")
```

استخدم وسيط "--num_processes" لتحديد عدد وحدات معالجة الرسومات (GPU) التي تريد استخدامها، ثم استدعي "accelerate launch" لتشغيل البرنامج النصي:

```bash
accelerate launch run_distributed.py --num_processes=2
```

### وضع الجهاز

مع Accelerate، يمكنك استخدام "device_map" لتحديد كيفية توزيع نماذج خط الأنابيب عبر أجهزة متعددة. وهذا مفيد في الحالات التي تحتوي على أكثر من وحدة معالجة الرسومات (GPU).

على سبيل المثال، إذا كان لديك وحدتي معالجة رسومات (GPU) بحجم 8 جيجابايت، فقد لا يعمل استخدام "DiffusionPipeline.enable_model_cpu_offload" بشكل جيد لأنه:

- يعمل فقط على وحدة معالجة رسومات (GPU) واحدة
- قد لا يتسع نموذج واحد على وحدة معالجة رسومات (GPU) واحدة ("DiffusionPipeline.enable_sequential_cpu_offload" قد يعمل ولكنه سيكون بطيئًا للغاية وهو أيضًا محدود بوحدة معالجة رسومات (GPU) واحدة)

للاستفادة من وحدتي معالجة الرسومات (GPU)، يمكنك استخدام استراتيجية "المتوازنة" (balanced) لوضع الجهاز والتي تقوم بتقسيم النماذج عبر جميع وحدات معالجة الرسومات (GPU) المتاحة.

قم باستدعاء "DiffusionPipeline.reset_device_map" لإعادة تعيين "device_map" لخط الأنابيب. وهذا ضروري أيضًا إذا كنت تريد استخدام طرق مثل "to()" و"DiffusionPipeline.enable_sequential_cpu_offload" و"DiffusionPipeline.enable_model_cpu_offload" على خط أنابيب تم تعيين جهازه.

```py
pipeline.reset_device_map()
```

بمجرد تعيين جهاز خط الأنابيب، يمكنك أيضًا الوصول إلى خريطة الجهاز الخاصة به عبر "hf_device_map":

```py
print(pipeline.hf_device_map)
```

قد تبدو خريطة الجهاز على النحو التالي:

```bash
{"unet": 1، "vae": 1، "safety_checker": 0، "text_encoder": 0}
```

### PyTorch Distributed

تدعم PyTorch "DistributedDataParallel" الذي يمكّن الموازاة بين البيانات.

للبدء، قم بإنشاء ملف Python واستورد "torch.distributed" و"torch.multiprocessing" لإعداد مجموعة العمليات الموزعة ولإنشاء العمليات للاستدلال على كل وحدة معالجة رسومات (GPU). يجب عليك أيضًا تهيئة "DiffusionPipeline":

```py
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from diffusers import DiffusionPipeline

sd = DiffusionPipeline.from_pretrained(
"runwayml/stable-diffusion-v1-5"، torch_dtype=torch.float16، use_safetensors=True
)
```

ستحتاج إلى إنشاء دالة لتشغيل الاستدلال؛ وتتعامل "init_process_group" مع إنشاء بيئة موزعة مع نوع backend الذي سيتم استخدامه، و"الرتبة" (rank) للعملية الحالية، و"حجم العالم" (world_size) أو عدد العمليات المشاركة. إذا كنت تقوم بتشغيل الاستدلال بالتوازي على وحدتي معالجة رسومات (GPU)، فإن "حجم العالم" (world_size) هو 2.

قم بنقل "DiffusionPipeline" إلى "الرتبة" (rank) واستخدم "get_rank" لتعيين وحدة معالجة الرسومات (GPU) لكل عملية، حيث تتعامل كل عملية مع موجه مختلف:

```py
def run_inference(الرتبة، حجم العالم):
dist.init_process_group ("nccl"، الرتبة=الرتبة، العالم_حجم=حجم العالم)

sd.to (الرتبة)

إذا كان torch.distributed.get_rank () == 0:
موجه = "كلب"
إلا إذا كان torch.distributed.get_rank () == 1:
موجه = "قطة"

الصورة = sd (موجه).images [0]
الصورة. save (f ". / {'_'.join (موجه)}. png")
```

لتشغيل الاستدلال الموزع، استدع "mp.spawn" لتشغيل دالة "run_inference" على عدد وحدات معالجة الرسومات (GPU) المحددة في "حجم العالم" (world_size):

```py
def main():
world_size = 2
mp.spawn(run_inference, args=(world_size،)، nprocs=world_size، join=True)


if __name__ == "__main__":
main()
```

بمجرد الانتهاء من كتابة البرنامج النصي للاستدلال، استخدم وسيط "--nproc_per_node" لتحديد عدد وحدات معالجة الرسومات (GPU) التي تريد استخدامها، ثم استدع "torchrun" لتشغيل البرنامج النصي:

```bash
torchrun run_distributed.py --nproc_per_node=2
```