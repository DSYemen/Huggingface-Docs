# تدريب PyTorch على Apple silicon 

في السابق، كان تدريب النماذج على أجهزة Mac يقتصر على وحدة المعالجة المركزية فقط. مع إصدار PyTorch v1.12، يمكنك الاستفادة من تدريب النماذج باستخدام معالجات الرسوميات Silicon GPUs من Apple للحصول على أداء وتدريب أسرع بشكل ملحوظ. يتم تشغيل هذا في PyTorch عن طريق دمج معالجات الرسوميات Metal Performance Shaders (MPS) من Apple كخلفية. تقوم خلفية [MPS](https://pytorch.org/docs/stable/notes/mps.html) بتنفيذ عمليات PyTorch كشظايا معدنية مخصصة وتضع هذه الوحدات النمطية على جهاز 'mps'. 

<Tip warning={true}> 

لم يتم بعد تنفيذ بعض عمليات PyTorch في MPS وقد يؤدي ذلك إلى حدوث خطأ. لتجنب ذلك، يجب عليك تعيين متغير البيئة `PYTORCH_ENABLE_MPS_FALLBACK=1` لاستخدام نوى وحدة المعالجة المركزية بدلاً من ذلك (ستظل ترى `UserWarning`). 

<br> 

إذا واجهتك أي أخطاء أخرى، يرجى فتح مشكلة في مستودع [PyTorch](https://github.com/pytorch/pytorch/issues) لأن [`Trainer`] يقوم بتكامل خلفية MPS فقط. 

</Tip> 

مع تعيين جهاز 'mps'، يمكنك: 

* تدريب الشبكات الأكبر أو أحجام الدفعات المحلية 
* تقليل زمن استرداد البيانات لأن بنية الذاكرة الموحدة لوحدة معالجة الرسومات تسمح بالوصول المباشر إلى مخزن الذاكرة الكامل 
* تقليل التكاليف لأنك لست بحاجة إلى التدريب على وحدات معالجة الرسومات السحابية أو إضافة وحدات معالجة رسومات محلية إضافية 

ابدأ بالتأكد من تثبيت PyTorch. يتم دعم تسريع MPS على macOS 12.3+. 

```bash
pip install torch torchvision torchaudio
``` 

يستخدم [`TrainingArguments`] جهاز 'mps' بشكل افتراضي إذا كان متاحًا، مما يعني أنك لست بحاجة إلى تعيين الجهاز بشكل صريح. على سبيل المثال، يمكنك تشغيل نص [run_glue.py](https://github.com/huggingface/transformers/blob/main/examples/pytorch/text-classification/run_glue.py) البرمجي مع تمكين خلفية MPS تلقائيًا دون إجراء أي تغييرات. 

```diff
export TASK_NAME=mrpc

python examples/pytorch/text-classification/run_glue.py \
--model_name_or_path google-bert/bert-base-cased \
--task_name $TASK_NAME \
- --use_mps_device \
--do_train \
--do_eval \
--max_seq_length 128 \
--per_device_train_batch_size 32 \
--learning_rate 2e-5 \
--num_train_epochs 3 \
--output_dir /tmp/$TASK_NAME/ \
--overwrite_output_dir
``` 

لا تدعم أجهزة 'mps' الخلفيات الخاصة بالتكوينات الموزعة مثل 'gloo' و'nccl'، مما يعني أنه يمكنك التدريب على وحدة معالجة رسومات واحدة فقط مع خلفية MPS. 

يمكنك معرفة المزيد عن خلفية MPS في منشور المدونة [تقديم تدريب PyTorch المعجل على Mac](https://pytorch.org/blog/introducing-accelerated-pytorch-training-on-mac/).