# علم بيرت 

هناك مجال متنامي من الدراسة يهتم بالتحقيق في آلية عمل المحولات الضخمة مثل BERT (والذي يطلق عليه البعض اسم "BERTology"). وفيما يلي بعض الأمثلة الجيدة على هذا المجال:

- BERT Rediscovers the Classical NLP Pipeline بقلم Ian Tenney وDipanjan Das وEllie Pavlick: https://arxiv.org/abs/1905.05950

- Are Sixteen Heads Really Better than One? بقلم Paul Michel وOmer Levy وGraham Neubig: https://arxiv.org/abs/1905.10650

- What Does BERT Look At? An Analysis of BERT's Attention بقلم Kevin Clark وUrvashi Khandelwal وOmer Levy وChristopher D. Manning: https://arxiv.org/abs/1906.04341

- CAT-probing: A Metric-based Approach to Interpret How Pre-trained Models for Programming Language Attend Code Structure: https://arxiv.org/abs/2210.04633

وللمساعدة في تطوير هذا المجال الجديد، قمنا بإضافة بعض الميزات الإضافية في نماذج BERT/GPT/GPT-2 للسماح للناس بالوصول إلى التمثيلات الداخلية، والتي تم تكييفها بشكل أساسي من العمل الرائع لـ Paul Michel (https://arxiv.org/abs/1905.10650):

- الوصول إلى جميع المخفيّات في BERT/GPT/GPT-2.

- الوصول إلى جميع أوزان الانتباه لكل رأس في BERT/GPT/GPT-2.

- استرجاع قيم ومدرجات إخراج الرؤوس لحساب درجة أهمية الرأس وإزالة الرؤوس غير المهمة كما هو موضح في https://arxiv.org/abs/1905.10650.

ولمساعدتك على فهم واستخدام هذه الميزات، قمنا بإضافة مثال نصي محدد: [bertology.py](https://github.com/huggingface/transformers/tree/main/examples/research_projects/bertology/run_bertology.py) والذي يقوم باستخراج المعلومات وإزالة فروع من نموذج مدرب مسبقًا على GLUE.