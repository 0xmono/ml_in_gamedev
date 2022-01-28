# Machine Learning Applications in Game Development

<details>
<summary>Table of Contents</summary>
<!-- MarkdownTOC -->

- [About](#about)
- [Common](#common)
- [Production](#production)
    - [Game](#game)
        - [Render](#render)
            - [Super Resolution](#super-resolution)
            - [Neural Rendering](#neural-rendering)
        - [Gameplay](#gameplay)
            - [Character](#character)
                - [Character Movement and Animation](#character-movement-and-animation)
            - [Facial Animation](#facial-animation)
            - [Ubisoft projects on movement and animations](#ubisoft-projects-on-movement-and-animations)
        - [Physics](#physics)
            - [Physics-based Simulation Group in TUM](#physics-based-simulation-group-in-tum)
                - [Physics-based Deep Learning Book](#physics-based-deep-learning-book)
                - [PhiFlow](#phiflow)
            - [Florian Marquardt courses](#florian-marquardt-courses)
        - [Multiplayer](#multiplayer)
            - [Prevent cheating/fraud/harm](#prevent-cheatingfraudharm)
            - [Net](#net)
            - [Prediction and smoothing](#prediction-and-smoothing)
            - [Matchmaking](#matchmaking)
        - [\(in-game\) AI](#in-game-ai)
            - [FIFA](#fifa)
        - [Text](#text)
        - [Controls](#controls)
            - [Pose](#pose)
            - [Voice](#voice)
        - [User-generated content](#user-generated-content)
            - [Avatars](#avatars)
- [Development](#development)
    - [Game Design](#game-design)
    - [Content creation](#content-creation)
        - [Art](#art)
            - [Art Search](#art-search)
            - [Images](#images)
                - [Multimodal-to-Image](#multimodal-to-image)
                - [Image-to-Image](#image-to-image)
                    - [Image-to-Image Supervised](#image-to-image-supervised)
                    - [Image-to-Image Unsupervised](#image-to-image-unsupervised)
                - [Textures](#textures)
                - [Generative](#generative)
                - [Text-to-Image](#text-to-image)
                    - [DALL-E](#dall-e)
                    - [CLIP](#clip)
                    - [Deep Daze](#deep-daze)
                    - [Diffusion Models](#diffusion-models)
                    - [Other](#other)
                - [Bitmaps](#bitmaps)
            - [3D Models](#3d-models)
                - [Reconstruction From Images and Video](#reconstruction-from-images-and-video)
                    - [Face Image](#face-image)
            - [Shape](#shape)
            - [Level](#level)
            - [Scene](#scene)
                - [From Image](#from-image)
            - [Voxel](#voxel)
            - [Cards](#cards)
                - [MTG](#mtg)
        - [Text](#text-1)
        - [Audio](#audio)
            - [Music](#music)
                - [Music Source Separation](#music-source-separation)
                - [Music Generation](#music-generation)
                - [Music Stryle Transfer](#music-stryle-transfer)
            - [Voice](#voice-1)
                - [TTS](#tts)
                    - [Soft](#soft)
            - [Game Commenting](#game-commenting)
            - [Cources](#cources)
        - [Video: trailers/cinematics](#video-trailerscinematics)
            - [Face Video](#face-video)
    - [Code](#code)
        - [Code Generation](#code-generation)
        - [Finding Bugs](#finding-bugs)
            - [Ubisof](#ubisof)
    - [Testing](#testing)
        - [Bots Playing Game](#bots-playing-game)
        - [Graphics](#graphics)
- [Game analytics](#game-analytics)
- [Common Models/Papers/Repos](#common-modelspapersrepos)
    - [DeepMind](#deepmind)
        - [EfficientZero](#efficientzero)
        - [MuZero](#muzero)
        - [AplhaZero](#aplhazero)
        - [AlphaGo](#alphago)
        - [DeepMind Environments](#deepmind-environments)
    - [Language Models](#language-models)
    - [Hugging Face Transformers](#hugging-face-transformers)
    - [Speech recognition](#speech-recognition)
    - [Classification](#classification)
    - [Detection and segmentation](#detection-and-segmentation)
        - [Detectron](#detectron)
    - [Tracking](#tracking)
    - [Depth Estimation](#depth-estimation)
    - [Action](#action)
    - [Human](#human)
        - [Datasets](#datasets)
        - [Action](#action-1)
        - [Clothes](#clothes)
            - [Virtual Try-on \(VTON\)](#virtual-try-on-vton)
        - [Person Pose and Shape](#person-pose-and-shape)
            - [Person Pose Detection](#person-pose-detection)
            - [Person Segmentation](#person-segmentation)
            - [Person Part Segmentation](#person-part-segmentation)
            - [Person Pose Estimation](#person-pose-estimation)
            - [Hand Pose Estimation](#hand-pose-estimation)
            - [Head Pose Estimation](#head-pose-estimation)
            - [Person Shape Capture](#person-shape-capture)
            - [Person Dense Pose](#person-dense-pose)
            - [Pose Retargeting](#pose-retargeting)
            - [Human-Object Interaction](#human-object-interaction)
            - [Only 2D](#only-2d)
        - [Statistical Body Models](#statistical-body-models)
        - [Motion](#motion)
            - [Motion Manifold](#motion-manifold)
        - [Human Synthesys](#human-synthesys)
            - [Head](#head)
    - [View Synthesis](#view-synthesis)
    - [Radiance Fields](#radiance-fields)
    - [Siren](#siren)
    - [Image Animation](#image-animation)
    - [Tools](#tools)
        - [Environments](#environments)
            - [OpenAI Environments](#openai-environments)
            - [Unity ML-Agents](#unity-ml-agents)
                - [AnimalAI](#animalai)
            - [Unreal Engine](#unreal-engine)
                - [Contests Environments](#contests-environments)
            - [Physics Environments](#physics-environments)
                - [MuJoCo Physics](#mujoco-physics)
        - [Libs](#libs)
        - [Render](#render-1)
        - [Deployment](#deployment)
        - [Optimization](#optimization)
        - [Vision transformer](#vision-transformer)

<!-- /MarkdownTOC -->
</details>

# About

This is a list of some existing and (mostly) possible applications of machine learning in game development: 
reseach projects, free/propriatory software, youtube videos and other.

It is just a (relatively) structured compilation of notes and not pretends to be precise and comprehensive information source on the topic. 
The goal is to have single-document list of short references to projects, videos, articles, books and other applications of ML in gamedev.

If you found an error or have something to add - feel free to create a pull request or open an issue.

<details>
<summary>Some designations in this document</summary>

<img src="https://upload.wikimedia.org/wikipedia/commons/d/df/Container_01_KMJ.jpg" alt="container" title="Contains references to multiple projects" width="30" height="30"/> - 
this image will be added to show that some project is a container of multiple references to another projects/software/dataset/books and so on. 
Usually it is a page on github, similar to this document, to some extent. 
See the [full image](https://upload.wikimedia.org/wikipedia/commons/d/df/Container_01_KMJ.jpg).

Typical names for web links in this document:

- <a name="project_link"></a> [[project](#project_link)] = main page of (reseach) project
- <a name="code_link"></a> [[code](#code_link)] = link to code, usually on [github](https://github.com)
- <a name="colab_link"></a> [[colab](#colab_link)] = link to [colab](https://colab.research.google.com)
- <a name="data_link"></a> [[data](#data_link)] = link to dataset or another data
- <a name="demo_link"></a> [[demo](#demo_link)] = link to work example of project/software
- <a name="paper_link"></a> [[paper](#paper_link)] = link to pdf paper itself or web page with link to it
- <a name="arxiv_link"></a> [[arxiv](#arxiv_link)] = link to publication page on [arxiv.org](https://arxiv.org)
- <a name="supplement_link"></a> [[supplement](#supplement_link)] = link to some (usually) conference-specific project page, containing supplementary material
- <a name="blog_link"></a> [[blog](#blog_link)] = some blog entry or article
- <a name="video_link"></a> [[video](#video_link)] = link to video, usually on [youtube](https://youtube.com)
- <a name="book_link"></a> [[book](#book_link)] = book online or offline
- <a name="soft_link"></a> [[soft](#soft_link)] = software product, possibly paid
- <a name="course_link"></a> [[course](#course_link)] = course of lectures on the topic
- <a name="wiki_link"></a> [[wiki](#wiki_link)] = link to wikipedia
- <a name="local_link"></a> [[local](#local_link)] = local link inside this document

</details>

# Common

Machine learning in video games 
[[wiki](https://en.wikipedia.org/wiki/Machine_learning_in_video_games)].

AI and Gaming Research Summit 2021 
[[video](https://www.youtube.com/watch?v=Ex3pJaunie0&list=PLD7HFcN7LXReQICXw1p2cV4fZHohdqX34)] - 
list of 12 videos.

The Alchemy and Science of Machine Learning for Games (GDC 2019)
[[video](https://www.youtube.com/watch?v=Eim_0jCQW_g)].

Machine Learning for Game Developers (Google I/O'19) 
[[video](https://www.youtube.com/watch?v=2h-Wg5FDbtU)].

Deep Learning for Game Developers (GDC 2018) 
[[video](https://www.youtube.com/watch?v=rF6Usm0tdhk)].

TAISIG Talks: Pieter Spronck on artificial intelligence in games 
[[video](https://www.youtube.com/watch?v=zhC8poMM2v4)]

AI FOR AI Game Challenge 2021 - Introduction to Machine Learning in Game Development 
[[video](https://www.youtube.com/watch?v=rkbrEk6TohA)].

Art Direction Bootcamp: Building a Creative Future with Artificial Intelligence
[[video](https://www.youtube.com/watch?v=9FAXAgRrOSE)]

<a name="MatthewGuzdialPCG"></a>
Game AI
[[video](https://www.youtube.com/playlist?list=PLSpVCA8HBelvTgCCKU1KHzwzsbgmV8QDi)] - 
part of [Matthew Guzdial](http://guzdial.com)'s University of Alberta class CMPUT 296: Game Artificial Intelligence cource lectures, 
covers mostly ```procedural content generation```[[local](#content-creation)] (PCG).

AI in Game industry: Building New Worlds and New Mindsets
[[video](https://www.youtube.com/watch?v=u2i87NT-MXo)] - talk from manager from SEED, Electronic Arts.

# Production

Machine learning functionality to run in production.

:warning: **Not all code and tools are redy for production!** Most is just for reference, to play around with it.

## Game

### Render

#### Super Resolution

Looks like super resolution will be a part of graphical API soon, but for now we have the following.

Nvidia DLSS
[[project](https://www.nvidia.com/en-us/geforce/technologies/dlss)]
[[code](https://github.com/NVIDIA/DLSS)]
[[download page](https://developer.nvidia.com/dlss-getting-started)]
[[Unreal Engine Plugin](https://developer.nvidia.com/dlss-getting-started)]
[[Unity: Supported in Unity 2021.2](https://github.com/Unity-Technologies/Graphics/pull/3484)].

GPUOpen(AMD) FidelityFX
[[project](https://www.amd.com/en/technologies/radeon-software-fidelityfx-super-resolution)]
[[code](https://github.com/GPUOpen-Effects/FidelityFX-FSR)]
[[Unreal Engine Plugin](https://gpuopen.com/learn/ue4-fsr)]
[[Unity: Supported in Unity 2021.2](https://github.com/Unity-Technologies/Graphics/pull/5152)].

Intel Architecture Day 2021 Demo: Xe HPG – High Quality Super Sampling 
[[video](https://www.youtube.com/watch?v=AH8g-wnc7Jo)] - 
to be released with Intel Arc GPUs.

DirectML Super Resolution 
[[code](https://github.com/microsoft/DirectML/tree/master/Samples/DirectMLSuperResolution)] - 
not for realtime upscaling.

Deep Learning Super Sampling using GANs 
[[code](https://github.com/vee-upatising/DLSS)] - 
not for realtime upscaling.

Neural Enhance
[[code](https://github.com/alexjc/neural-enhance)].

#### Neural Rendering

<img src="https://upload.wikimedia.org/wikipedia/commons/d/df/Container_01_KMJ.jpg" alt="container" title="Contains references to multiple projects" width="30" height="30"/>
Awesome neural rendering
[[project](https://github.com/weihaox/awesome-neural-rendering)] - 
github repo with collection of resources on neural rendering.

GANcraft: Unsupervised 3D Neural Rendering of Minecraft Worlds (ICCV 2021)
[[project](https://nvlabs.github.io/GANcraft)]
[[code](https://github.com/NVlabs/imaginaire/blob/master/projects/gancraft/README.md)]
[[arxiv](https://arxiv.org/abs/2104.07659)]
[[blog](https://developer.nvidia.com/blog/gancraft-turning-gamers-into-3d-artists)]
[[video](https://www.youtube.com/watch?v=1Hky092CGFQ)]
[[video](https://www.youtube.com/watch?v=5K-AgDmCtt0)] - 
part of ```Imaginaire```[[local](#Imaginaire)].
Made with help of ```Mineways```[[project](https://www.realtimerendering.com/erich/minecraft/public/mineways)].

Neural 3D Mesh Renderer (CVPR 2018)
[[project](https://hiroharu-kato.com/publication/neural_renderer)]
[[code](https://github.com/hiroharu-kato/neural_renderer)]
[[code](https://github.com/daniilidis-group/neural_renderer)]
[[arxiv](https://arxiv.org/abs/1711.07566)]
[[video](https://www.youtube.com/watch?v=vziVsrCaMHY)].

Advances in Neural Rendering 
[[arxiv](https://arxiv.org/abs/2111.05849)] - 
state-of-the-art report on advances in neural rendering on 10 Nov 2021.

Advances in Neural Rendering SIGGRAPH 2021 
[[course](https://www.neuralrender.com)] - 
SIGGRAPH 2021 Course on Advances in Neural Rendering. 
Consists of two videos: 
[[video (part1)](https://www.youtube.com/watch?v=otly9jcZ0Jg)]
[[video (part2)](https://www.youtube.com/watch?v=aboFl5ozImM)].

Advances in Neural Rendering CVPR 2020 [[course](https://www.neuralrender.com/CVPR)] - 
CVPR 2020 Course on Advances in Neural Rendering. 
Consists of two videos: 
[[video morning session](https://www.youtube.com/watch?v=LCTYRqW-ne8)] and 
[[video afternoon session](https://www.youtube.com/watch?v=JlyGNvbGKB8)].

MIT 6.S191 (2020): Neural Rendering [[video](https://www.youtube.com/watch?v=BCZ56MU-KhQ)] - single lecture from MIT `Introduction to
Deep Learning` course.

Matthias Niessner - Why Neural Rendering is Super Cool!
[[video](https://www.youtube.com/watch?v=-KGZmzP4P1I)].

ADL4CV - Neural rendering
[[video](https://www.youtube.com/watch?v=yh4BHFGUx70)].

See ```View Synthesis```[[local](#view-synthesis)].

See ```Radiance Fields```[[local](#radiance-fields)].

### Gameplay

#### Character

Talking To AI-Generated People \| Fake Faces, Script, Voice and Lip-Sync Animation
[[code](https://github.com/ChintanTrivedi/ask-fake-ai-karen)]
[[blog](https://medium.com/swlh/how-to-create-fake-talking-head-videos-with-deep-learning-code-tutorial-f9cfc0c19ab5)]
[[video](https://www.youtube.com/watch?v=OCdikmAoLKA)]
[[video](https://www.youtube.com/watch?v=zA8Qs8G5Vnc)].

##### Character Movement and Animation

<img src="https://upload.wikimedia.org/wikipedia/commons/d/df/Container_01_KMJ.jpg" alt="container" title="Contains references to multiple projects" width="30" height="30"/>
AI4Animation [[project](https://github.com/sebastianstarke/AI4Animation)] - 
[Sebastian Starke's](https://github.com/sebastianstarke) 
collection of deep learning  opportunities for character animation and control. 
Some projects from this collection are listed below.
*TODO*: add papers from here.

<img src="https://upload.wikimedia.org/wikipedia/commons/d/df/Container_01_KMJ.jpg" alt="container" title="Contains references to multiple projects" width="30" height="30"/>
Daniel Holden's cite [[project](https://theorangeduck.com)] - 
with numerous publications and articles about machine learning and animation.
*TODO*: add papers from here.

Learning Time-Critical Responses for Interactive Character Control
[[project](https://mrl.snu.ac.kr/research/ProjectAgile/Agile.html)]
[[code](https://github.com/snumrl/TimeCriticalResponse)]
[[paper](http://mrl.snu.ac.kr/research/ProjectAgile/AGILE_2021_SIGGRAPH_author.pdf)]
[[video](https://www.youtube.com/watch?v=rQKuvxg5ZHc)].

Transition Motion Tensor: A Data-Driven Approach for Versatile and Controllable Agents in Physically Simulated Environments (SIGGRAPH 2021) 
[[project](https://inventec-ai-center.github.io/projects/TMT_2021/index.html)]
[[code](https://github.com/inventec-ai-center/transition_motion_tensor)]
[[arxiv](https://arxiv.org/abs/2111.15072)]
[[video](https://www.youtube.com/watch?v=9NzRSZyAOiY)].

<a name="AdversarialMotionPriors"></a>
AMP: Adversarial Motion Priors for Stylized Physics-Based Character Control (SIGGRAPH 2021)
[[project](https://xbpeng.github.io/projects/AMP/index.html)]
[[code](https://github.com/xbpeng/DeepMimic)]
[[arxiv](https://arxiv.org/abs/2104.02180)]
[[video](https://www.youtube.com/watch?v=wySUxZN_KbM)]
[[video](https://www.youtube.com/watch?v=O6fBSMxThR4)].
Shares code with ```DeepMimic```[[local](#DeepMimic)].

A Scalable Approach to Control Diverse Behaviors for Physically Simulated Characters (SIGGRAPH 2020)
[[code](https://github.com/facebookresearch/ScaDiver)]
[[paper](https://research.facebook.com/publications/a-scalable-approach-to-control-diverse-behaviors-for-physically-simulated-characters)]
[[supplement](https://dl.acm.org/doi/abs/10.1145/3386569.3392381)]
[[video](https://www.youtube.com/watch?v=QnIwwAKX5H4)].

CARL: Controllable Agent with Reinforcement Learning for Quadruped Locomotion (SIGGRAPH 2020)
[[project](https://inventec-ai-center.github.io/projects/CARL/index.html)]
[[code](https://github.com/inventec-ai-center/carl-siggraph2020)]
[[arxiv](https://arxiv.org/abs/2005.03288)]
[[video](https://www.youtube.com/watch?v=t9CdF_Pl19Q)].

Character Controllers using Motion VAEs (SIGGRAPH 2020)
[[project](https://hungyuling.com/projects/MVAE)]
[[demo](https://github.com/electronicarts/character-motion-vaes)]
[[demo](https://github.com/belinghy/MotionVAEs-WebGL)]
[[paper](https://hungyuling.com/static/projects/MVAE/2020-MVAE.pdf)]
[[arxiv](https://arxiv.org/abs/2103.14274)].

MCP: Learning Composable Hierarchical Control with Multiplicative Compositional Policies (NeurIPS 2019)
[[project](https://xbpeng.github.io/projects/MCP)]
[[arxiv](https://arxiv.org/abs/1905.09808)].

Learning Predict-and-Simulate Policies From Unorganized Human Motion Data (SIGGRAPH Asia 2019)
[[project](https://mrl.snu.ac.kr/publications/ProjectICC/ICC.html)]
[[code](https://github.com/snumrl/ICC)]
[[paper](https://mrl.snu.ac.kr/publications/ProjectICC/ICC.pdf)]
[[video](https://www.youtube.com/watch?v=9dgUMli0HFU)].

<a name="DeepMimic"></a>
DeepMimic: Example-Guided Deep Reinforcement Learning of Physics-Based Character Skills (SIGGRAPH 2018)
[[project](https://xbpeng.github.io/projects/DeepMimic/index.html)]
[[code](https://github.com/xbpeng/DeepMimic)]
[[arxiv](https://arxiv.org/abs/1804.02717)].
Shares code with ```AMP: Adversarial Motion Priors``` [[local](#AdversarialMotionPriors)].

Mode-Adaptive Neural Networks for Quadruped Motion Control (SIGGRAPH 2018)
[[project](https://www.starke-consult.de/portfolio/assets/content/work/11/page.html)]
[[code (AI4Animation)](https://github.com/sebastianstarke/AI4Animation/tree/master/AI4Animation/SIGGRAPH_2018)]
[[code](https://github.com/cghezhang/MANN)]
[[data](http://www.starke-consult.de/AI4Animation/SIGGRAPH_2018/MotionCapture.zip)]
[[demo (windows)](http://www.starke-consult.de/AI4Animation/SIGGRAPH_2018/Demo_Windows.zip)]
[[demo (linux)](http://www.starke-consult.de/AI4Animation/SIGGRAPH_2018/Demo_Linux.zip)]
[[demo (mac)](http://www.starke-consult.de/AI4Animation/SIGGRAPH_2018/Demo_Mac.zip)]
[[paper](https://homepages.inf.ed.ac.uk/tkomura/dog.pdf)]
[[paper](https://github.com/sebastianstarke/AI4Animation//blob/master/Media/SIGGRAPH_2018/Paper.pdf)]
[[video](https://www.youtube.com/watch?v=uFJvRYtjQ4c)]
[[video](https://www.youtube.com/watch?v=55MBKxIHHYA)].

Deep Learning of Biomimetic Sensorimotor Control for Biomechanical Human Animation (SIGGRAPH 2018)
[[project](https://tomerwei.github.io/Deep%20Learning%20of%20Biomimetic%20Sensorimotor%20Control%20for%20Biomechanical%20Human%20Animation.html)]
[[paper](https://tomerwei.github.io/pdfs/nakada2018.pdf)]
[[video](https://www.youtube.com/watch?v=oh2ExRftTIc)].

Deep Learning For Animation & Content Creation (GDC 2018)
[[video](https://www.youtube.com/watch?v=nFk_-alrrxQ)].

Phase-Functioned Neural Networks for Character Control (SIGGRAPH 2017)
[[project](https://theorangeduck.com/page/phase-functioned-neural-networks-character-control)]
[[code](https://github.com/sreyafrancis/PFNN)]
[[code (AI4Animation)](https://github.com/sebastianstarke/AI4Animation/blob/master/AI4Animation/SIGGRAPH_2017)]
[[code (+ data, zip array)](http://theorangeduck.com/media/uploads/other_stuff/pfnn.zip)]
[[demo](http://theorangeduck.com/media/uploads/other_stuff/pfnn_demo.zip)]
[[demo (windows)](http://www.starke-consult.de/AI4Animation/SIGGRAPH_2017/Demo_Windows.zip)]
[[demo (linux)](http://www.starke-consult.de/AI4Animation/SIGGRAPH_2017/Demo_Linux.zip)]
[[demo (mac)](http://www.starke-consult.de/AI4Animation/SIGGRAPH_2017/Demo_Mac.zip)]
[[paper](https://theorangeduck.com/media/uploads/other_stuff/phasefunction.pdf)]
[[video](https://www.youtube.com/watch?v=Ul0Gilv5wvY)]
[[video (GDC 2018)](https://www.youtube.com/watch?v=o-QLSjSSyVk)].

DeepLoco: Dynamic Locomotion Skills Using Hierarchical Deep Reinforcement Learning (SIGGRAPH 2017)
[[project](https://www.cs.ubc.ca/~van/papers/2017-TOG-deepLoco/index.html)]
[[project](https://xbpeng.github.io/projects/DeepLoco/index.html)]
[[code](https://github.com/xbpeng/DeepLoco)]
[[paper](https://www.cs.ubc.ca/~van/papers/2017-TOG-deepLoco/2017-TOG-deepLoco.pdf)]
[[supplement](https://www.cs.ubc.ca/~van/papers/2017-TOG-deepLoco/2017-TOG-deepLoco-supp.pdf)]
[[video](https://www.youtube.com/watch?v=G4lT9CLyCNw)]
[[video](https://www.youtube.com/watch?v=hd1yvLWm6oA)]
[[video](https://www.youtube.com/watch?v=x-HrYko_MRU)].

Emergence of Locomotion Behaviours in Rich Environments (2017)
[[arxiv](https://arxiv.org/abs/1707.02286)]
[[blog](https://deepmind.com/blog/article/producing-flexible-behaviours-simulated-environments)]
[[video](https://www.youtube.com/watch?v=hx_bgoTF7bs)].

Learning human behaviors from motion capture by adversarial imitation (2017)
[[data](http://mocap.cs.cmu.edu/)]
[[arxiv](https://arxiv.org/abs/1707.02201)]
[[blog](https://deepmind.com/blog/article/producing-flexible-behaviours-simulated-environments)]
[[video](https://www.youtube.com/watch?v=YsxN3uRBupc)].

Robust Imitation of Diverse Behaviors (2017)
[[arxiv](https://arxiv.org/pdf/1707.02747.pdf)]
[[blog](https://deepmind.com/blog/article/producing-flexible-behaviours-simulated-environments)]
[[video](https://www.youtube.com/watch?v=NaohsyUxpxw)]
[[video](https://www.youtube.com/watch?v=VBrIll0B24o)].

Terrain-Adaptive Locomotion Skills Using Deep Reinforcement Learning (SIGGRAPH 2016)
[[project](https://www.cs.ubc.ca/~van/papers/2016-TOG-deepRL/index.html)]
[[code](https://github.com/xbpeng/DeepTerrainRL)]
[[paper](https://www.cs.ubc.ca/~van/papers/2016-TOG-deepRL/2016-TOG-deepRL.pdf)]
[[video](https://www.youtube.com/watch?v=KPfzRSBzNX4)]
[[video](https://www.youtube.com/watch?v=A0BmHoujP9k)].

Guided Learning of Control Graphs for Physics-Based Characters (SIGGRAPH 2016)
[[project](https://www.cs.ubc.ca/~van/papers/2016-TOG-controlGraphs/index.html)]
[[paper](https://www.cs.ubc.ca/~van/papers/2016-TOG-controlGraphs/2016-TOG-controlGraphs.pdf)].

Flexible Muscle-Based Locomotion for Bipedal Creatures (SIGGRAPH Asia 2013)
[[project](https://www.goatstream.com/research/papers/SA2013)]
[[paper](https://www.goatstream.com/research/papers/SA2013/SA2013.pdf)]
[[video](https://www.youtube.com/watch?v=pgaEE27nsQw)].

See ```Motion```[[local](#motion)].

#### Facial Animation

The Eyes Have It: An Integrated Eye and Face Model for Photorealistic Facial Animation (2020)
[[paper](https://dl.acm.org/doi/pdf/10.1145/3386569.3392493)]
[[blog](https://research.facebook.com/publications/the-eyes-have-it-an-integrated-eye-and-face-model-for-photorealistic-facial-animation)]
[[video](https://www.youtube.com/watch?v=E3spq4pS2Y4)].

Audio- and Gaze-driven Facial Animation of Codec Avatars (2020)
[[project](https://research.facebook.com/videos/audio-and-gaze-driven-facial-animation-of-codec-avatars)]
[[paper](Audio- and Gaze-driven Facial Animation of Codec Avatars)]
[[arxiv](https://arxiv.org/abs/2008.05023)]
[[blog](https://medium.com/deepgamingai/realistic-facial-animations-of-3d-avatars-driven-by-audio-and-gaze-9be2102e24d)]
[[video](https://www.youtube.com/watch?v=1nZjW_xoCDQ)].

VisemeNet: Audio-Driven Animator-Centric Speech Animation
[[project](https://people.umass.edu/~yangzhou/visemenet)]
[[code](https://github.com/yzhou359/VisemeNet_tensorflow)]
[[paper](https://people.umass.edu/~yangzhou/visemenet/visemenet.pdf)]
[[arxiv](https://arxiv.org/abs/1805.09488)]
[[video](https://www.youtube.com/watch?v=kk2EnyMD3mo)].

Audio-Driven Facial Animation by Joint End-to-End Learning of Pose and Emotion (SIGGRAPH 2017)
[[paper](https://research.nvidia.com/publication/2017-07_Audio-Driven-Facial-Animation)]
[[code (unafficial)](https://github.com/leventt/surat)]
[[video](https://www.youtube.com/watch?v=lDzrfdpGqw4)].

NVIDIA Omniverse Audio2Face App
[[soft](https://www.nvidia.com/en-us/omniverse/apps/audio2face)].

Speech2Face
[[code](https://github.com/saiteja-talluri/Speech2Face)]
[[arxiv](https://arxiv.org/abs/1905.09773)].

#### Ubisoft projects on movement and animations

SuperTrack: Motion Tracking for Physically Simulated Characters using Supervised Learning (SIGGRAPH Asia 2021)
[[paper](https://static-wordpress.akamaized.net/montreal.ubisoft.com/wp-content/uploads/2021/11/24183638/SuperTrack.pdf)]
[[blog](https://montreal.ubisoft.com/en/supertrack-motion-tracking-for-physically-simulated-characters-using-supervised-learning)]
[[video](https://www.youtube.com/watch?v=8sMjfGkQ4bw)].

Learned Motion Matching (SIGGRAPH 2020)
[[paper](https://static-wordpress.akamaized.net/montreal.ubisoft.com/wp-content/uploads/2020/07/09154101/Learned_Motion_Matching.pdf)]
[[blog](https://montreal.ubisoft.com/en/introducing-learned-motion-matching)]
[[video](https://www.youtube.com/watch?v=16CHDQK4W5k)].

Robust Motion In-betweening (SIGGRAPH 2020)
[[data](https://github.com/ubisoft/ubisoft-laforge-animation-dataset)]
[[paper](https://static-wordpress.akamaized.net/montreal.ubisoft.com/wp-content/uploads/2020/07/09155337/RobustMotionInbetweening.pdf)]
[[blog](https://montreal.ubisoft.com/en/automatic-in-betweening-for-faster-animation-authoring)]
[[video](https://www.youtube.com/watch?v=fTV7sXqO6ig)].

Machine Learning Summit: Ragdoll Motion Matching (GDC 2020)
[[project](https://www.gdcvault.com/play/1026712/Machine-Learning-Summit-Ragdoll-Motion)]
[[video](https://www.youtube.com/watch?v=JZKaqQKcAnw)]
[[video](https://www.youtube.com/watch?v=lN9pXZzR3Ys)]. 
About motion matching itself
[[supplement](https://www.gdcvault.com/play/1023115/Animation-Bootcamp-Motion-Matching-The)]
[[video](https://www.youtube.com/watch?v=JZKaqQKcAnw)]
[[video](https://www.youtube.com/watch?v=KSTn3ePDt50)]
[[video](https://www.youtube.com/watch?v=z_wpgHFSWss)].

DReCon: Data-Driven responsive Control of Physics-Based Characters (SIGGRAPH Asia 2019)
[[paper](https://static-wordpress.akamaized.net/montreal.ubisoft.com/wp-content/uploads/2019/11/13214229/DReCon.pdf)]
[[blog](https://montreal.ubisoft.com/en/drecon-data-driven-responsive-control-of-physics-based-characters)]
[[video](https://www.youtube.com/watch?v=tsdzwmEGl2o)].

Robust Solving of Optical Motion Capture Data by Denoising
[[paper](https://montreal.ubisoft.com/wp-content/uploads/2018/05/neuraltracker.pdf)]
[[blog](https://montreal.ubisoft.com/en/robust-solving-of-optical-motion-capture-data-by-denoising)].

### Physics

SimGAN: Hybrid Simulator Identification for Domain Adaptation via Adversarial Reinforcement Learning (ICRA 2021)
[[code](https://github.com/jyf588/SimGAN)]
[[arxiv](https://arxiv.org/abs/2101.06005)]
[[blog](https://ai.googleblog.com/2021/06/learning-accurate-physics-simulator-via.html)]
[[video](https://www.youtube.com/watch?v=McKOGllO7nc)].

Subspace Neural Physics: Fast Data-Driven Interactive Simulation (GDC 2020)
[[paper](https://theorangeduck.com/media/uploads/other_stuff/deep-cloth-paper.pdf)]
[[blog](https://theorangeduck.com/page/subspace-neural-physics-fast-data-driven-interactive-simulation)]
[[blog](https://theorangeduck.com/page/machine-learning-kolmogorov-complexity-squishy-bunnies)]
[[blog](https://montreal.ubisoft.com/en/ubisoft-la-forge-produces-a-data-driven-physics-simulation-based-on-machine-learning)]
[[video](https://www.youtube.com/watch?v=yjEvV86byxg)]
[[video (GDC talk)](https://www.youtube.com/watch?v=sUb0W5_waRI)].

Learning to Simulate Complex Physics with Graph Networks (ICML 2020)
[[project](https://sites.google.com/view/learning-to-simulate)]
[[code](https://github.com/deepmind/deepmind-research/tree/master/learning_to_simulate)]
[[arxiv](https://arxiv.org/abs/2002.09405)]
[[supplement](https://icml.cc/virtual/2020/poster/6849)].

Use the Force, Luke! Learning to Predict Physical Forces by Simulating Effects (CVPR 2020)
[[project](https://ehsanik.github.io/forcecvpr2020)]
[[code](https://github.com/ehsanik/touchTorch)]
[[arxiv](https://arxiv.org/abs/2003.12045)]
[[slides](https://github.com/ehsanik/forcecvpr2020/blob/master/img/3095-talk.pdf)]
[[video](https://ehsanik.github.io/forcecvpr2020/#slide_video)]
[[video](https://www.youtube.com/watch?v=dx3_nXcOqV0)].

<img src="https://upload.wikimedia.org/wikipedia/commons/d/df/Container_01_KMJ.jpg" alt="container" title="Contains references to multiple projects" width="30" height="30"/>
[Jie Tan's Homepage](https://www.jie-tan.net/publication.html) - 
ML applied to robots movement, but looks like can be used in game development too.

See ```Physics```[[local](#physics-environments)].

#### Physics-based Simulation Group in TUM

<img src="https://upload.wikimedia.org/wikipedia/commons/d/df/Container_01_KMJ.jpg" alt="container" title="Contains references to multiple projects" width="30" height="30"/>
[Physics-based Simulation Group](https://ge.in.tum.de) in 
[Technical University of Munich](https://www.tum.de/en) - 
has multiple publications, see their cite and links below.

<img src="https://upload.wikimedia.org/wikipedia/commons/d/df/Container_01_KMJ.jpg" alt="container" title="Contains references to multiple projects" width="30" height="30"/>
Physics-Based Deep Learning
[[project](https://github.com/thunil/Physics-Based-Deep-Learning)] - 
github pahe with ```A LOT``` of links to papers!

##### Physics-based Deep Learning Book

Physics-based Deep Learning Book
[[book](https://physicsbaseddeeplearning.org)] - 
great online book with possiblity to run some example code in 
[[colab](https://colab.research.google.com/github/tum-pbs/pbdl-book/blob/main)] and 
[source](https://github.com/tum-pbs/pbdl-book) on gihub.

Introducing the Physics-based Deep Learning book (PBDL) 
[[video](https://www.youtube.com/watch?v=SU-OILSmR1M)].

##### PhiFlow

PhiFlow is a research-oriented, differentiable, open-source physics simulation toolkit 
[[code](https://github.com/tum-pbs/PhiFlow)].

phiflow2blender Tutorial
[[code](https://github.com/intergalactic-mammoth/phiflow2blender)]
[[video](https://www.youtube.com/watch?v=xI1ARz4ZSQU)].

PhiFlow Tutorials
[[video](https://www.youtube.com/playlist?list=PLYLhRkuWBmZ5R6hYzusA2JBIUPFEE755O)] playlist.

Differentiable Physics (for Deep Learning), Overview Talk by 
[Nils Thuerey](https://ge.in.tum.de/about/n-thuerey)
[[video](https://www.youtube.com/watch?v=BwuRTpTR2Rg)], 
another similar talk
[[video](https://www.youtube.com/watch?v=fKYX3xoZn6c)].

#### Florian Marquardt courses

Machine Learning for Physicists
[[video](https://www.youtube.com/watch?v=qMp3s7D_8Xw&list=PLemsnf33Vij4eFWwtoQCrt9AHjLe3uo9_)] 
[[course](https://pad.gwdg.de/Machine_Learning_For_Physicists_2020)] - 
Florian Marquardt's course is not for game development, but anyway good course connecting ML and simulation.

Advanced Machine Learning for Physics, Science, and Artificial Scientific Discovery
[[video](https://www.youtube.com/watch?v=B2Jnurp-OkU&list=PLemsnf33Vij4-kv-JTjDthaGUYUnQbbws)]
[[course](https://pad.gwdg.de/s/2021_AdvancedMachineLearningForScience)] - 
the next course.

### Multiplayer

#### Prevent cheating/fraud/harm

Robust Vision-Based Cheat Detection in Competitive Gaming
[[arxiv](https://arxiv.org/abs/2103.10031)]
[[blog](https://research.nvidia.com/publication/2021-03_Robust-Vision-Based-Cheat)]
[[video](https://youtu.be/si1omc3T9W4?t=4016)].

Using an Artificial Neural Network to detect aim assistance in Counter-Strike: Global Offensive
[[paper](https://www.cs.nmt.edu/~kmaberry/ann_fps_cheater.pdf)].

Here's how we're using AI to help detect misinformation
[[blog](https://ai.facebook.com/blog/heres-how-were-using-ai-to-help-detect-misinformation)].

Harmful content can evolve quickly. Our new AI system adapts to tackle it
[[blog](https://about.fb.com/news/2021/12/metas-new-ai-system-tackles-harmful-content)]
[[blog](https://ai.facebook.com/blog/harmful-content-can-evolve-quickly-our-new-ai-system-adapts-to-tackle-it)].

Robocalypse Now: Using Deep Learning to Combat Cheating in Counter-Strike: Global Offensive (GDC 2018)
[[video](https://www.youtube.com/watch?v=kTiP0zKF9bc)].

See ```Language Models``` [[local](#language-models)].

#### Net

*Possible*: better compression, based on values boudns and probabilities collected for every game map/mode, etc.

#### Prediction and smoothing

*Possible*: better replicated entities parameters prediction and smoothing based on learned players behaviour.

#### Matchmaking

The Wanderings of Odysseus in 3D Scenes (2021)
[[project](https://yz-cnsdqz.github.io/eigenmotion/GAMMA)]
[[arxiv](https://arxiv.org/abs/2112.09251)].

Machine Learning for Optimal Matchmaking (GDC 2020)
[[video](https://www.youtube.com/watch?v=JG155gDdhrE)], 
similar talk Machine learning for optimal matchmaking (Game Stack Live) [[video](https://www.youtube.com/watch?v=JG155gDdhrE)].

OptMatch: Optimized Matchmaking via Modeling the High-Order Interactions on the Arena (KDD2020)
[[code](https://github.com/fuxiAIlab/OptMatch)] - is it used somewhere?

Globally Optimized Matchmaking in Online Games (KDD2021)
[[code](https://github.com/fuxiAIlab/GloMatch)]
[[paper](https://dl.acm.org/doi/10.1145/3447548.3467074)] - paper and code to be published..

### (in-game) AI

Applying Reinforcement Learning to Develop Game AI in NetEase Games (2020 GDC)
[[video](https://www.youtube.com/watch?v=gXilr5C9MZs)].

Using Neural Networks to Control Agent Threat Response
[[paper](https://www.gameaipro.com/GameAIPro/GameAIPro_Chapter30_Using_Neural_Networks_to_Control_Agent_Threat_Response.pdf)].

See ```DeepMind and based on their research papers and repos``` [[local](#deepmind)].

#### FIFA

DeepGamingAI_FIFA
[[code](https://github.com/ChintanTrivedi/DeepGamingAI_FIFA)]
[[blog](https://medium.com/@chintan.t93/building-a-deep-neural-network-to-play-fifa-18-dce54d45e675)]
[[video](https://www.youtube.com/watch?v=vZFNzwv61Fk&t=59s)] - 
deep learning based AI bot for playing the football simulation game.

DeepGamingAI_FIFARL (2018)
[[code](https://github.com/ChintanTrivedi/DeepGamingAI_FIFARL)]
[[blog](https://towardsdatascience.com/using-deep-q-learning-in-fifa-18-to-perfect-the-art-of-free-kicks-f2e4e979ee66)]
[[video](https://www.youtube.com/watch?v=MasxAN-xZIU)] - 
using Reinforcement Learning to play FIFA.

### Text

*Possible*: adaptive in-game texsts like dialogues, descriptions etc.

See ```Language Models``` [[local](#language-models)].

### Controls

#### Pose

Possible using pose as a control input, (relatively) similar to [Kinect](https://en.wikipedia.org/wiki/Kinect).

See ```Person Pose Estimation```[[local](#person-pose-estimation)].

See ```Detectron```[[local](#detectron)].

See ```Motion```[[local](#motion)].

Playing Mortal Kombat with TensorFlow.js. Transfer learning and data augmentation (2018)
[[code](https://github.com/mgechev/movement.js)]
[[code](https://github.com/mgechev/mk.js)]
[[blog](https://blog.mgechev.com/2018/10/20/transfer-learning-tensorflow-js-data-augmentation-mobile-net)]
[[video](https://www.youtube.com/watch?v=0_yfU_iNUYo)].

#### Voice

See ```Speech Recognition``` [[local](#speech-recognition)].

### User-generated content

#### Avatars

DECA: Learning an Animatable Detailed 3D Face Model from In-The-Wild Images (SIGGRAPH2021)
[[project](https://deca.is.tue.mpg.de)]
[[project](https://ps.is.mpg.de/publications/feng-siggraph-2021)]
[[code](https://github.com/YadiraF/DECA)]
[[paper](https://files.is.tue.mpg.de/black/papers/SIGGRAPH21_DECA.pdf)]
[[arxiv](https://arxiv.org/abs/2012.04012)]
[[supplement](https://files.is.tue.mpg.de/black/papers/SIGGRAPH21_DECA_SupMat.pdf)]
[[video](https://www.youtube.com/watch?v=Yo53Q0N6N4M)].

<a name="Synthesia"></a>
Synthesia
[[soft](https://www.synthesia.io)] - 
creates ai videos of speaking avatars.

# Development

Machine learning functionality to help with game development.

How Ubisoft La Forge Integrates Machine Learning into Game Production
[[blog](https://80.lv/articles/how-ubisoft-la-forge-integrates-machine-learning-into-game-production)].

## Game Design

Automated Game Design via Conceptual Expansion
[[arxiv](https://arxiv.org/abs/1809.02232)]
[[blog](https://thenewstack.io/ai-automates-video-game-design-with-conceptual-expansion)].

## Content creation

AI4CC: AI for Content Creation Workshop (CVPR 2021)
[[project](https://visual.cs.brown.edu/workshops/aicc2021)]
[[video](https://www.youtube.com/watch?v=x6cHOulDXUo&list=PLNPyQ_mnkEr7ae_dbhv4MG2ja5EI74lzu)].

AI for Content Creation Workshop (CVPR 2020)
[[project](https://visual.cs.brown.edu/workshops/aicc2020)]
[[video (morning session)](https://www.youtube.com/watch?v=z8jbOo9EFsI)]
[[video (afternoon session)](https://www.youtube.com/watch?v=DrizewlvZGc)]
[[video (paper session 1)](https://www.youtube.com/watch?v=xqOpYIht_1I)]
[[video (paper session 2)](https://www.youtube.com/watch?v=JsGINhvaJAY)].

Friend, Collaborator, Student, Manager: How Design of an AI-Driven Game Level Editor Affects Creators (CHI '19)
[[arxiv](https://arxiv.org/abs/1901.06417)]
[[supplement](https://dl.acm.org/doi/10.1145/3290605.3300854)].

Deep Learning for Procedural Content Generation
[[arxiv](https://arxiv.org/abs/2010.04548)].

PCGML: Procedural Content Generation via Machine Learning
[[arxiv](https://arxiv.org/abs/1702.00539)].

See Matthew Guzdial's course videos [[local](#MatthewGuzdialPCG)].

Julian Togelius - Increasing Generality in Reinforcement Learning through PCG @ UCL DARK
[[video](https://www.youtube.com/watch?v=9KPcUgnjpMg)].

### Art

De-rendering the World's Revolutionary Artefacts (CVPR 2021)
[[project](https://sorderender.github.io)]
[[code](https://github.com/elliottwu/sorderender)]
[[arxiv](https://arxiv.org/abs/2104.03954)]
[[video](https://www.youtube.com/watch?v=pxkYyyw02H0)] - 
learns to de-render a single image into shape, albedo and complex lighting and material components, allowing for novel-view synthesis and relighting.

#### Art Search

Promethean Ai
[[soft](https://www.prometheanai.com)]
[[video](https://www.youtube.com/watch?v=73ZTnPsO-m0)]
[[video](https://www.youtube.com/watch?v=hA0MsGWvmzs)] - 
"clever" art browser.

#### Images

A lot of possibilities for generation images such as concept art or textures.

MMEditing
[[code](https://github.com/open-mmlab/mmediting)] - 
open source image and video editing toolbox based on PyTorch.
*TODO*: add pertinent models from MMEditing here explicitly.

See ```View Synthesis```[[local](#view-synthesis)].

See ```Radiance Fields```[[local](#radiance-fields)].

##### Multimodal-to-Image

Multimodal Conditional Image Synthesis with Product-of-Experts GANs (2021)
[[project](https://deepimagination.cc/PoE-GAN)]
[[arxiv](https://arxiv.org/abs/2112.05130)]
[[video](https://www.youtube.com/watch?v=56aA_FaeAPY)] - 
text + sketch + segmentation = image.

##### Image-to-Image

<img src="https://upload.wikimedia.org/wikipedia/commons/d/df/Container_01_KMJ.jpg" alt="container" title="Contains references to multiple projects" width="30" height="30"/>
awesome image-to-image translation
[[project](https://github.com/weihaox/awesome-image-translation)]
*TODO*: add papers from here.

###### Image-to-Image Supervised

SPADE: Semantic Image Synthesis with Spatially-Adaptive Normalization (CVPR 2019)
[[project](https://nvlabs.github.io/SPADE)]
[[code](https://github.com/NVlabs/imaginaire/blob/master/projects/spade/README.md)]
[[code (Previous Implementation)](https://github.com/NVlabs/SPADE)]
[[arxiv](https://arxiv.org/abs/1903.07291)]
[[video](https://www.youtube.com/watch?v=p5U4NgVGAwg)]
[[video](https://www.youtube.com/watch?v=MXWm6w4E5q0)] - 
part of ```Imaginaire```[[local](#Imaginaire)].

pix2pixHD: High-Resolution Image Synthesis and Semantic Manipulation with Conditional GANs (CVPR 2018)
[[project](https://tcwang0509.github.io/pix2pixHD)]
[[code](https://github.com/NVlabs/imaginaire/blob/master/projects/pix2pixhd/README.md)]
[[code (Previous Implementation)](https://github.com/NVIDIA/pix2pixHD)]
[[arxiv](https://arxiv.org/abs/1711.11585)]
[[video](https://www.youtube.com/watch?v=3AIpPlzM_qs)] - 
part of ```Imaginaire```[[local](#Imaginaire)].

Image-to-Image Translation with Conditional Adversarial Nets (CVPR 2017)
[[project](https://phillipi.github.io/pix2pix)]
[[code](https://github.com/phillipi/pix2pix)]
[[code (PyTorch)](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)]
[[colab](https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/generative/pix2pix.ipynb)]
[[colab](https://github.com/tensorflow/docs/blob/master/site/en/tutorials/generative/pix2pix.ipynb)]
[[demo](https://affinelayer.com/pixsrv)]
[[arxiv](https://arxiv.org/abs/1611.07004)]
[[blog](https://affinelayer.com/pix2pix)]
[[blog](https://ml4a.github.io/guides/Pix2Pix)]
[[blog](https://machinelearningmastery.com/a-gentle-introduction-to-pix2pix-generative-adversarial-network)]
[[video (Two Minute Papers)](https://www.youtube.com/watch?v=u7kQ5lNfUfg)].

###### Image-to-Image Unsupervised

Rethinking the Truly Unsupervised Image-to-Image Translation (ICCV 2021)
[[code](https://github.com/clovaai/tunit)]
[[arxiv](https://arxiv.org/abs/2006.06500)]
[[video](https://www.youtube.com/watch?v=sEG8hD64c_Q)].

Few-Shot Unsupervised Image-to-Image Translation on complex scenes
[[arxiv](https://arxiv.org/abs/2106.03770)].

COCO-FUNIT: Few-Shot Unsupervised Image Translation with a Content Conditioned Style Encoder (ECCV 2020)
[[project](https://nvlabs.github.io/COCO-FUNIT)]
[[code](https://github.com/NVlabs/imaginaire/blob/master/projects/coco_funit/README.md)]
[[paper](https://nvlabs.github.io/COCO-FUNIT/paper.pdf)]
[[video](https://www.youtube.com/watch?v=btnDfqcedrk)]
[[video](https://www.youtube.com/watch?v=Ewfx2Um75aw)].

DUNIT: Detection-based Unsupervised Image-to-Image Translation (CVPR 2020)
[[code](https://github.com/IVRL/Dunit)]
[[paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Bhattacharjee_DUNIT_Detection-Based_Unsupervised_Image-to-Image_Translation_CVPR_2020_paper.pdf)]
[[video](https://www.youtube.com/watch?v=ENK1ROiZPms)].

StarGAN v2: Diverse Image Synthesis for Multiple Domains (CVPR 2020)
[[code](https://github.com/clovaai/stargan-v2)]
[[arxiv](https://arxiv.org/abs/1912.01865)]
[[video](https://www.youtube.com/watch?v=0EVh5Ki4dIY)]
[[video](https://www.youtube.com/watch?v=sx5x4KGqX6s)].

FUNIT: Few-Shot Unsupervised Image-to-Image Translation (ICCV 2019)
[[project](https://nvlabs.github.io/FUNIT)]
[[code](https://github.com/NVlabs/imaginaire/blob/master/projects/funit/README.md)]
[[code](https://github.com/nvlabs/FUNIT)]
[[arxiv](https://arxiv.org/abs/1905.01723)]
[[video](https://www.youtube.com/watch?v=kgPAqsC8PLM)] - 
part of ```Imaginaire```[[local](#Imaginaire)].

Recent Advances in Unsupervised Image-to-Image Translation (2019)
[[video](https://www.youtube.com/watch?v=NsPMlDsRCkM)].

MUNIT: Multimodal Unsupervised Image-to-Image Translation (ECCV 2018)
[[code](https://github.com/NVlabs/imaginaire/blob/master/projects/munit/README.md)]
[[code (Previous Implementation)](https://github.com/NVlabs/MUNIT)]
[[arxiv](https://arxiv.org/abs/1804.04732)]
[[video](https://www.youtube.com/watch?v=ab64TWzWn40)] - 
part of ```Imaginaire```[[local](#Imaginaire)].

CVPR18: Session 3-3A: Machine Learning for Computer Vision V
[[video](https://www.youtube.com/watch?v=sIkUzmgUaxc)] - 
including ```StarGAN```[[local](#StarGAN)].

<a name="StarGAN"></a>
StarGAN: Unified Generative Adversarial Networks for Multi-Domain Image-to-Image Translation (CVPR 2018)
[[code](https://github.com/yunjey/stargan)]
[[arxiv](https://arxiv.org/abs/1711.09020)]
[[video](https://www.youtube.com/watch?v=EYjdLppmERE)]
[[video](https://www.youtube.com/watch?v=sIkUzmgUaxc)]
[[video](https://www.youtube.com/watch?v=8XfcDkkFbMs)]
[[video](https://www.youtube.com/watch?v=lw1lUCreJ0k)].

UNIT: Unsupervised Image-to-Image Translation (NeurIPS 2017)
[[code](https://github.com/NVlabs/imaginaire/blob/master/projects/unit/README.md)]
[[code (Previous Implementation)](https://github.com/mingyuliutw/UNIT)]
[[paper](https://proceedings.neurips.cc/paper/2017/file/dc6a6489640ca02b0d42dabeb8e46bb7-Paper.pdf)]
[[arxiv](https://arxiv.org/abs/1703.00848)]
[[video](https://www.youtube.com/watch?v=nlyXoX2aIek)] - 
part of ```Imaginaire```[[local](#Imaginaire)].

BicycleGAN: Toward Multimodal Image-to-Image Translation
[[project](https://junyanz.github.io/BicycleGAN)]
[[code](https://github.com/junyanz/BicycleGAN)]
[[arxiv](https://arxiv.org/abs/1711.11586)]
[[poster](https://junyanz.github.io/BicycleGAN/index_files/poster_nips_v3.pdf)]
[[video](https://www.youtube.com/watch?v=JvGysD2EFhw)].

Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks (ICCV 2017)
[[project](https://junyanz.github.io/CycleGAN)]
[[code](https://github.com/junyanz/CycleGAN)]
[[code (PyTorch)](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)]
[[colab](https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/generative/cyclegan.ipynb)]
[[colab](https://github.com/tensorflow/docs/blob/master/site/en/tutorials/generative/cyclegan.ipynb)]
[[arxiv](https://arxiv.org/abs/1703.10593)]
[[blog](https://hardikbansal.github.io/CycleGANBlog)]
[[blog](https://machinelearningmastery.com/what-is-cyclegan)]
[[video](https://www.youtube.com/watch?v=AxrKVfjSBiA)].

##### Textures

Generative Modelling of BRDF Textures from Flash Images (SIGGRAPH Asia 2021)
[[project](https://henzler.github.io/publication/neuralmaterial)]
[[code](https://github.com/henzler/neuralmaterial)]
[[data](https://drive.google.com/drive/folders/1uUTTyWGM75lkTP0RaawPTALD1eZZ-AXq?usp=sharing)]
[[arxiv](https://arxiv.org/abs/2102.11861)]
[[video](https://www.youtube.com/watch?v=_HTiiKxccJ4)].

Implicit Feature Networks for Texture Completion from Partial 3D Data (ECCV 2020)
[[project](https://virtualhumans.mpi-inf.mpg.de/ifnets)]
[[code](https://github.com/jchibane/if-net_texture)]
[[paper](https://virtualhumans.mpi-inf.mpg.de/papers/jchibane20ifnet/SHARP2020.pdf)]
[[arxiv](https://arxiv.org/abs/2009.09458)].

Learning a Neural 3D Texture Space from 2D Exemplars (CVPR 2020)
[[project](https://geometry.cs.ucl.ac.uk/projects/2020/neuraltexture)]
[[code](https://github.com/henzler/neuraltexture)]
[[paper](https://geometry.cs.ucl.ac.uk/projects/2020/neuraltexture/paper_docs/neuraltexture.pdf)]
[[arxiv](https://arxiv.org/abs/1912.04158)]
[[video](https://www.youtube.com/watch?v=it5y2qaONBE)].

Appearance Controlled Face Texture Generation for Video Games Characters
[[video](https://www.youtube.com/watch?v=ykaTVnG_z1w)].

##### Generative

MMGeneration
[[code](https://github.com/open-mmlab/mmgeneration)] - open-source toolkit for generative models, especially for GANs now, based on PyTorch.
*TODO*: add pertinent models from MMGeneration here explicitly.

Lecture 13 \| Generative Models
[[video](https://www.youtube.com/watch?v=5WoItGTWV54)] - 
Stanford course ```CS231n: Convolutional Neural Networks for Visual Recognition```
[[course](http://cs231n.stanford.edu)].

##### Text-to-Image

Improving Text-to-Image Synthesis Using Contrastive Learning (BMVC 2021)
[[arxiv](https://arxiv.org/abs/2107.02423)].

DF-GAN: Deep Fusion Generative Adversarial Networks for Text-to-Image Synthesis
[[code](https://github.com/tobran/DF-GAN)]
[[arxiv](https://arxiv.org/abs/2008.05865)].
*TODO*: see projects referenced in github repo.

###### DALL-E

DALL·E: Creating Images from Text
[[code](https://github.com/openai/dall-e)]
[[arxiv](https://arxiv.org/abs/2102.12092)]
[[blog](https://openai.com/blog/dall-e)].

OpenAI DALL·E: Creating Images from Text (Blog Post Explained)
[[video](https://www.youtube.com/watch?v=j4xgkjWlfL4)].

DALL-E in Pytorch
[[code](https://github.com/lucidrains/DALLE-pytorch)].

ruDALL-E
[[code](https://github.com/sberbank-ai/ru-dalle)] - generate images from texts in russian.

DALL-E и CLIP от OpenAI Новая эпоха в машинном обучении [ru] / Михаил Константинов
[[video (russian)](https://www.youtube.com/watch?v=FUAuMiyFFtE)].

###### CLIP

CLIP: Learning Transferable Visual Models From Natural Language Supervision 
[[code](https://github.com/openai/CLIP)]
[[arxiv](https://arxiv.org/abs/2103.00020)]
[[blog](https://openai.com/blog/clip)].

CLIP
[[blog](https://huggingface.co/docs/transformers/model_doc/clip)] - 
CLIP model documentation on ```Hugging Face```[[local](#hugging-face-transformers)].

OpenAI's CLIP is the most important advancement in computer vision this year
[[blog](https://blog.roboflow.com/openai-clip)].

OpenAI CLIP: ConnectingText and Images (Paper Explained)
[[video](https://www.youtube.com/watch?v=T9XSU0pKX2E)].

OpenAI CLIP - Connecting Text and Images \| Paper Explained
[[video](https://www.youtube.com/watch?v=fQyHEXZB-nM)].

OpenAI’s CLIP explained! \| Examples, links to code and pretrained model
[[video](https://www.youtube.com/watch?v=dh8Rxhf7cLU)].

CLIP: Connecting Text and Images
[[video](https://www.youtube.com/watch?v=u0HG77RNhPE0)].

###### Deep Daze

Deep Daze
[[code](https://github.com/lucidrains/deep-daze)] - 
simple command line tool for text to image generation using 
```OpenAI's CLIP``` [[local](#clip)] and 
```Siren```[[local](#siren)].

###### Diffusion Models

GLIDE: Towards Photorealistic Image Generation and Editing with Text-Guided Diffusion Models 
[[code](https://github.com/openai/glide-text2im)]
[[arxiv](https://arxiv.org/abs/2112.10741)]
[[video (explanation1)](https://www.youtube.com/watch?v=gwI6g1pBD84)]
[[video (explanation2)](https://www.youtube.com/watch?v=lvv4N2nf-HU)].

Diffusion Models Beat GANS on Image Synthesis
[[code](https://github.com/crowsonkb/guided-diffusion)]
[[arxiv](https://arxiv.org/abs/2105.05233)]. 
DDPM - Diffusion Models Beat GANs on Image Synthesis (Machine Learning Research Paper Explained)
[[video](https://www.youtube.com/watch?v=W-O7AZNzbzQ)]. 
Обзор статьи "Diffusion Models Beat GANs on Image Synthesis"
[[video (russian)](https://www.youtube.com/watch?app=desktop&v=WyWLu1MVkXY)].

Improved Denoising Diffusion Probabilistic Models
[[code](https://github.com/openai/improved-diffusion)]
[[arxiv](https://arxiv.org/abs/2102.09672)]
[[video](https://www.youtube.com/watch?v=g2VV26c1Tgo)].

Denoising Diffusion Probabilistic Models
[[project](https://hojonathanho.github.io/diffusion)]
[[code](https://github.com/hojonathanho/diffusion)]
[[arxiv](https://arxiv.org/abs/2006.11239)].

Denoising Diffusion Probabilistic Model, in Pytorch
[[code](https://github.com/lucidrains/denoising-diffusion-pytorch)].

v objective diffusion inference code for PyTorch
[[code](https://github.com/crowsonkb/v-diffusion-pytorch)] and JAX 
[[code](https://github.com/crowsonkb/v-diffusion-jax)].

A new SotA for generative modelling — Denoising Diffusion Probabilistic Models
[[blog](https://medium.com/graphcore/a-new-sota-for-generative-modelling-denoising-diffusion-probabilistic-models-8e21eec6792e)].

Autoregressive Diffusion Models
[[arxiv](https://arxiv.org/abs/2110.02037)].
Google Proposes ARDMs: Efficient Autoregressive Models That Learn to Generate in any Order
[[blog](https://medium.com/syncedreview/google-proposes-ardms-efficient-autoregressive-models-that-learn-to-generate-in-any-order-8acbf819b817)]. 
Autoregressive Diffusion Models (Machine Learning Research Paper Explained)
[[video](https://www.youtube.com/watch?v=2h4tRsQzipQ)].

###### Other

CogView: Mastering Text-to-Image Generation via Transformers
[[code](https://github.com/THUDM/CogView)]
[[arxiv](https://arxiv.org/abs/2105.13290)] - 
generate vivid Images for Any (Chinese) text.

VirTex: Learning Visual Representations from Textual Annotations
[[code](https://github.com/kdexd/virtex)]
[[arxiv](https://arxiv.org/abs/2006.06666)]
[[video](https://www.youtube.com/watch?v=L-Fx-7WqPPs)].

##### Bitmaps

<a name="WaveFunctionCollapse"></a>
WaveFunctionCollapse
[[code](https://github.com/mxgmn/WaveFunctionCollapse)] - 
this program generates bitmaps that are locally similar to the input bitmap.

Addressing the Fundamental Tension of PCGML with Discriminative Learning
[[arxiv](https://arxiv.org/abs/1809.04432)] - based on 
```WaveFunctionCollapse```[[local](#WaveFunctionCollapse)].

#### 3D Models

See ```3D Morphable Models```[[local](3DMM)].

##### Reconstruction From Images and Video

See ```Person Shape Capture```[[local](#person-shape-capture)].

Kimera
[[code](https://github.com/MIT-SPARK/Kimera)] - 
C++ library for real-time metric-semantic simultaneous localization and mapping, 
which uses camera images and inertial data to build a semantically annotated 3D mesh of the environment.

BANMo: Building Animatable 3D Neural Models from Many Casual Videos
[[project](https://banmo-www.github.io)]
[[arxiv](https://arxiv.org/abs/2112.12761)].

DOVE: Learning Deformable 3D Objects by Watching Videos
[[project](https://dove3d.github.io)]
[[arxiv](https://arxiv.org/abs/2107.10844)]
[[video](https://www.youtube.com/watch?v=_FsADb0XmpY)].

ViSER: Video-Specific Surface Embeddings for Articulated 3D Shape Reconstruction (NeurIPS 2021)
[[project](https://viser-shape.github.io)]
[[code](https://github.com/gengshan-y/viser-release)]
[[paper](https://www.contrib.andrew.cmu.edu/~gengshay/ViSER.pdf)].

NeRS: Neural Reflectance Surfaces for Sparse-View 3D Reconstruction in the Wild (NeurIPS 2021)
[[project](https://jasonyzhang.com/ners)]
[[code](https://github.com/jasonyzhang/ners)]
[[colab](https://colab.research.google.com/drive/1L4Sl_9Osc2J_I5YpkteLrb-VbnwdDokd?usp=sharing)]
[[arxiv](https://arxiv.org/abs/2110.07604)]
[[video](https://www.youtube.com/watch?v=zVyaw_sn1xM)].

Panoptic 3D Scene Reconstruction From a Single RGB Image (NeurIPS 2021)
[[project](https://manuel-dahnert.com/research/panoptic-reconstruction)]
[[code](https://github.com/xheon/panoptic-reconstruction)]
[[paper](https://proceedings.neurips.cc/paper/2021/file/46031b3d04dc90994ca317a7c55c4289-Paper.pdf)]
[[arxiv](https://arxiv.org/abs/2111.02444)]
[[video](https://www.youtube.com/watch?v=YVxRNHmd5SA)].

Ray-ONet: Efficient 3D Reconstruction From A Single RGB Image (BMVC 2021)
[[project](https://rayonet.active.vision)]
[[code](https://github.com/ActiveVisionLab/ray-onet)]
[[paper](https://www.bmvc2021-virtualconference.com/assets/papers/0698.pdf)]
[[arxiv](https://arxiv.org/abs/2107.01899)].

Pixel-Perfect Structure-from-Motion with Featuremetric Refinement (ICCV 2021)
[[project](https://psarlin.com/pixsfm)]
[[code](https://github.com/cvg/pixel-perfect-sfm)]
[[arxiv](https://arxiv.org/abs/2108.08291)]
[[supplement](https://psarlin.com/pixsfm/assets/pixsfm_slides.pdf)]
[[supplement](https://psarlin.com/pixsfm/assets/pixsfm_poster.pdf)]
[[video](https://www.youtube.com/watch?v=2HuCMuraFk0)].

Graduate School – Deep Learning on Meshes – Rana Hanocka (SGP 2021)
[[video](https://www.youtube.com/watch?v=qVctAmMGlQQ)].

Learning monocular 3D reconstruction of articulated categories from motion (CVPR 2021)
[[project](https://fkokkinos.github.io/video_3d_reconstruction)]
[[code](https://github.com/fkokkinos/acfm_video_3d_reconstruction)]
[[paper](https://fkokkinos.github.io/video_3d_reconstruction/resources/pdf/paper.pdf)]
[[arxiv](https://arxiv.org/abs/2103.16352)]
[[supplement](https://fkokkinos.github.io/video_3d_reconstruction/resources/pdf/supplementary.pdf)].

LASR: Learning Articulated Shape Reconstruction from a Monocular Video (CVPR 2021)
[[project](https://lasr-google.github.io)]
[[code](https://github.com/google/lasr)]
[[paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Yang_LASR_Learning_Articulated_Shape_Reconstruction_From_a_Monocular_Video_CVPR_2021_paper.pdf)]
[[blog](https://www.louisbouchard.ai/3d-reconstruction-from-videos)]
[[video](https://www.youtube.com/watch?v=Y6KGQeUKYAI)].

Unsupervised Learning of 3D Object Categories from Videos in the Wild (CVPR 2021)
[[project](https://henzler.github.io/publication/unsupervised_videos)]
[[arxiv](https://arxiv.org/abs/2103.16552)]
[[supplement](https://henzler.github.io/publication/unsupervised_videos/UnsupervisedVideosPoster.pdf)]
[[video](https://www.youtube.com/watch?v=910z84dldEU)].

Shelf-supervised Mesh Prediction in the wild (CVPR 2021)
[[project](https://judyye.github.io/ShSMesh)]
[[code](https://github.com/JudyYe/shelf-sup-mesh)]
[[arxiv](https://arxiv.org/abs/2102.06195)]
[[video](https://www.youtube.com/watch?v=OAiFEAuzPZk)].

Learning 3D Registration and Reconstruction from the Visual World
[[paper](https://chenhsuanlin.bitbucket.io/thesis.pdf)]
[[video](https://www.youtube.com/watch?v=7F73EIyuFI4)]
[[video](https://www.youtube.com/watch?v=xR_tCdNRHpo)] - 
doctoral thesis of [Chen-Hsuan Lin](https://chenhsuanlin.bitbucket.io).

To The Point: Correspondence-driven monocular 3D category reconstruction
[[project](https://fkokkinos.github.io/to_the_point)]
[[paper](https://fkokkinos.github.io/to_the_point/resources/pdf/paper.pdf)]
[[arxiv](https://arxiv.org/abs/2106.05662)].

SDF-SRN: Learning Signed Distance 3D Object Reconstruction from Static Images (NeurIPS 2020)
[[project](https://chenhsuanlin.bitbucket.io/signed-distance-SRN)]
[[code](https://github.com/chenhsuanlin/signed-distance-SRN)]
[[paper](https://chenhsuanlin.bitbucket.io/signed-distance-SRN/paper.pdf)]
[[arxiv](https://arxiv.org/abs/2010.10505)]
[[video](https://www.youtube.com/watch?v=7F73EIyuFI4)].

Self-supervised Single-view 3D Reconstruction via Semantic Consistency
[[project](https://sites.google.com/nvidia.com/unsup-mesh-2020)]
[[code](https://github.com/NVlabs/UMR)]
[[video](https://www.youtube.com/watch?v=vLUf-msmb3s)].

U-CMR: Shape and Viewpoint without Keypoints (ECCV 2020)
[[project](https://shubham-goel.github.io/ucmr)]
[[code](https://github.com/shubham-goel/ucmr)]
[[arxiv](https://arxiv.org/abs/2007.10982)]
[[video](https://www.youtube.com/watch?v=9-Ttb8jsevo)] - 
Unsupervised Category-Specific Mesh Reconstruction.

Articulation-Aware Canonical Surface Mapping (CVPR 2020)
[[project](https://nileshkulkarni.github.io/acsm)]
[[code](https://github.com/nileshkulkarni/acsm)]
[[arxiv](https://arxiv.org/abs/2004.00614)]
[[video](https://www.youtube.com/watch?v=hECMGIGGybA)].

Non-line-of-sight Surface Reconstruction Using the Directional Light-cone Transform (CVPR 2020)
[[project](https://www.computationalimaging.org/publications/nlos_dlct)]
[[code](https://github.com/computational-imaging/nlos-dlct)]
[[paper](https://www.computationalimaging.org/wp-content/uploads/2020/03/dlct_cvpr2020.pdf)]
[[supplement](https://www.computationalimaging.org/wp-content/uploads/2020/03/dlct_supplement_cvpr2020.pdf)]
[[video](https://www.youtube.com/watch?v=9ezA5ycHXDA)].

DISN: Deep Implicit Surface Network for High-quality Single-view 3D Reconstruction NeurIPS 2019)
[[code](https://github.com/Xharlie/DISN)]
[[code](https://github.com/laughtervv/DISN)]
[[data](https://github.com/Xharlie/ShapenetRender_more_variation)]
[[arxiv](https://arxiv.org/abs/1905.10711)].

Canonical Surface Mapping via Geometric Cycle Consistency (ICCV 2019)
[[project](https://nileshkulkarni.github.io/csm)]
[[code](https://github.com/nileshkulkarni/csm)]
[[arxiv](https://arxiv.org/abs/1907.10043)]
[[video](https://www.youtube.com/watch?v=93M3ou4mg-w)].

Escaping Plato's Cave: 3D Shape From Adversarial Rendering (ICCV2019)
[[project](https://henzler.github.io/publication/platonicgan)]
[[project](https://geometry.cs.ucl.ac.uk/projects/2019/platonicgan)]
[[code](https://github.com/henzler/platonicgan)]
[[paper](https://geometry.cs.ucl.ac.uk/projects/2019/platonicgan/paper_docs/platonicgan.pdf)]
[[arxiv](https://arxiv.org/abs/1811.11606)]
[[supplement](https://henzler.github.io/publication/platonicgan/poster.pdf)].

Learning to Reconstruct Shapes from Unseen Classes (NeurIPS 2018)
[[project](http://genre.csail.mit.edu)]
[[code](https://github.com/xiumingzhang/GenRe-ShapeHD)]
[[paper](http://genre.csail.mit.edu/papers/genre_nips.pdf)]
[[arxiv](https://arxiv.org/abs/1812.11166)]
[[supplement](http://genre.csail.mit.edu/papers/genre_nips_supp.pdf)]
[[video](https://www.youtube.com/watch?v=DA9KmoFGIXw)].

Learning Category-Specific Mesh Reconstruction from Image Collections (ECCV, 2018)
[[project](https://akanazawa.github.io/cmr)]
[[code](https://github.com/akanazawa/cmr)]
[[arxiv](https://arxiv.org/abs/1803.07549)]
[[video](https://www.youtube.com/watch?v=cYHQKtBLI3Q)].

Learning Shape Priors for Single-View 3D Completion and Reconstruction (ECCV 2018)
[[project](http://shapehd.csail.mit.edu)]
[[code](https://github.com/xiumingzhang/GenRe-ShapeHD)]
[[paper](http://shapehd.csail.mit.edu/papers/shapehd_eccv.pdf)]
[[arxiv](https://arxiv.org/abs/1809.05068)].

MarrNet: 3D Shape Reconstruction via 2.5D Sketches (NIPS 2017)
[[project](http://marrnet.csail.mit.edu)]
[[code](https://github.com/jiajunwu/marrnet)]
[[paper](https://jiajunwu.com/papers/marrnet_nips.pdf)]
[[arxiv](https://arxiv.org/abs/1711.03129)]
[[supplement](http://marrnet.csail.mit.edu/talks/marrnet_poster_nips.pdf)]
[[video](https://www.youtube.com/watch?v=wTnVVcPU0go)].

Learning a Probabilistic Latent Space of Object Shapes via 3D Generative-Adversarial Modeling (NIPS 2016)
[[project](http://3dgan.csail.mit.edu)]
[[code](https://github.com/zck119/3dgan-release)]
[[code (PyTorch)](https://github.com/black0017/3D-GAN-pytorch)]
[[code (Keras)](https://github.com/enochkan/3dgan-keras)]
[[data](http://3dgan.csail.mit.edu/data/IKEA_imgs_shapes.zip)]
[[paper](http://3dgan.csail.mit.edu/papers/3dgan_nips.pdf)]
[[arxiv](https://arxiv.org/abs/1610.07584)]
[[supplement](http://3dgan.csail.mit.edu/talks/3dgan_poster_nips.pdf)]
[[video](https://www.youtube.com/watch?v=mfx7uAkUtCI)].

Single Image 3D Interpreter Network (ECCV 2016), 
3D Interpreter Networks for Viewer-Centered Wireframe Modeling (IJCV 2018)
[[project](http://3dinterpreter.csail.mit.edu)]
[[code](https://github.com/jiajunwu/3dinn)]
[[paper (ECCV 2016)](http://3dinterpreter.csail.mit.edu/papers/3dinn_eccv.pdf)]
[[paper (IJCV 2018)](http://3dinterpreter.csail.mit.edu/papers/3dinn_ijcv.pdf)]
[[paper (ECCV 2016)](https://arxiv.org/abs/1604.08685)]
[[paper (IJCV 2018)](https://arxiv.org/abs/1804.00782)]
[[video (ECCV 2016)](https://videolectures.net/eccv2016_wu_single_image)].

Distinguished AI Lecture Series \| Humans, hands, and horses
[[video](https://www.youtube.com/watch?v=0kNFXL37xO0)].

CVPR2021 3D Scene Understanding Workshop
[[video](https://www.youtube.com/playlist?list=PL6QXpwvKhQIrSklNsQJqicjtxiOs6_NpY)] - 
list of 9 videos.

CVPR2020 3D Scene Understanding Workshop: Zoom Recording
[[video](https://www.youtube.com/playlist?list=PL6QXpwvKhQIodh0xusfVnUPkBYq0AJ4O_)]
[[video (invited speakers)](https://www.youtube.com/playlist?list=PL6QXpwvKhQIoYOLLv_zSbl9pJOlIM4RIp)]
[[video (invited papers)](https://www.youtube.com/playlist?list=PL6QXpwvKhQIp6i8wsmrTL8tSl1QcvQIeT)].

CVPR 2019 Oral Session 2-1B: 3D Single View & RGBD
[[vide](https://www.youtube.com/watch?v=ko6kNZ9DuAk)].

Tutorial : 3D Deep Learning (CVPR 2017)
[[video](https://www.youtube.com/watch?v=8CenT_4HWyY)].

See ```Radiance Fields```[[local](#radiance-fields)].

###### Face Image

MeInGame: Create a Game Character Face from a Single Portrait
[[code](https://github.com/FuxiCV/MeInGame)]
[[data](https://drive.google.com/file/d/1tSBHEQ06XjY1yFMe9EIkFz-euU1EurXv/view)]
[[arxiv](https://arxiv.org/abs/2102.02371)]
[[video](https://www.youtube.com/watch?v=597cvKOegfE)].

Accurate 3D Face Reconstruction with Weakly-Supervised Learning: From Single Image to Image Set
[[code](https://github.com/microsoft/Deep3DFaceReconstruction)]
[[code (PyTorch)](https://github.com/sicxu/Deep3DFaceRecon_pytorch)]
[[arxiv](https://arxiv.org/abs/1903.08527)].

#### Shape

Superquadrics Revisited: Learning 3D Shape Parsing beyond Cuboids (CVPR 2019)
[[project](https://superquadrics.com)]
[[project](https://avg.is.mpg.de/publications/liao2018cvpr)]
[[code](https://github.com/paschalidoud/superquadric_parsing)]
[[paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Paschalidou_Superquadrics_Revisited_Learning_3D_Shape_Parsing_Beyond_Cuboids_CVPR_2019_paper.pdf)]
[[arxiv](https://arxiv.org/abs/1904.09970)]
[[blog](https://autonomousvision.github.io/superquadrics-revisited)]
[[video](https://www.youtube.com/watch?v=eaZHYOsv9Lw)].

Learning Shape Abstractions by Assembling Volumetric Primitives (CVPR 2017)
[[project](https://shubhtuls.github.io/volumetricPrimitives)]
[[code](https://github.com/shubhtuls/volumetricPrimitives)]
[[code (PyTorch)](https://github.com/nileshkulkarni/volumetricPrimitivesPytorch)]
[[arxiv](https://arxiv.org/abs/1612.00404)].

#### Level

PCGRL: Procedural Content Generation via Reinforcement Learning
[[code](https://github.com/amidos2006/gym-pcgrl)]
[[arxiv](https://arxiv.org/abs/2001.09212)]
[[blog](https://medium.com/deepgamingai/game-level-design-with-reinforcement-learning-52b02bb94954)]
[[video](https://www.youtube.com/watch?v=ml3Y1ljVSQ8)].

Illuminating Diverse Neural Cellular Automata for Level Generation
[[code](https://github.com/smearle/control-pcgrl)]
[[arxiv](https://arxiv.org/abs/2109.05489)].

ATISS: Autoregressive Transformers for Indoor Scene Synthesis
[[code](https://github.com/nv-tlabs/ATISS)]
[[arxiv](https://arxiv.org/abs/2110.03675)].

Adversarial Reinforcement Learning for Procedural Content Generation (CoG 21)
[[arxiv](https://arxiv.org/abs/2103.04847)]
[[blog](https://www.ea.com/seed/news/cog2021-adversarial-rl-content-generation)]
[[video](https://www.youtube.com/watch?v=kNj0qcc6Fpg)]
[[video](https://video.itu.dk/video/71682895/linus-gisslen-andy-eakins-camilo)].

Toward Game Level Generation from Gameplay Videos
[[arxiv](https://arxiv.org/abs/1602.07721)].

Co-Creative Level Design via Machine Learning
[[arxiv](https://arxiv.org/abs/1809.09420)]
[[video](https://www.youtube.com/watch?v=UkMeM5Ty1lA)].

Sampling Hyrule: Sampling Probabilistic Machine Learning for Level Generation
[[paper](https://www.aaai.org/ocs/index.php/AIIDE/AIIDE15/paper/view/11570/11395)] - 
The Legend of Zelda.

Player Movement Models for Platformer Game Level Generation
[[paper](https://www.ijcai.org/proceedings/2017/0105.pdf)] - 
Super Mario Bros.

Tutorial: Using machine learning for Level Generation in Snake (video-game)
[[video](https://www.youtube.com/watch?v=pBhHvXyFi7Y)].

Ben Berman - Machine Learning and Level Generation
[[video](https://www.youtube.com/watch?v=Z6lHExfem6U)].

#### Scene

##### From Image

360-Dataset
[[data](https://vcl.iti.gr/360-dataset)]
[[arxiv](https://arxiv.org/abs/1807.09620)] - 
see papers below.

2nd OmniCV workshop (2021 CVPR)
[[project](https://sites.google.com/view/omnicv2021)]
[[video](https://www.youtube.com/watch?v=xa7Fl2mD4CA)].

360o Surface Regression with a Hyper-Sphere Loss (3DV 2019)
[[project](https://vcl3d.github.io/HyperSphereSurfaceRegression)]
[[code](https://github.com/VCL3D/HyperSphereSurfaceRegression)]
[[data](https://vcl3d.github.io/3D60)]
[[model](https://github.com/VCL3D/HyperSphereSurfaceRegression/releases/tag/1.0)]
[[arxiv](https://arxiv.org/abs/1909.07043)].

Layer-structured 3D Scene Inference via View Synthesis (ECCV 2018)
[[project](https://shubhtuls.github.io/lsi)]
[[code](https://github.com/google/layered-scene-inference)]
[[arxiv](https://arxiv.org/abs/1807.10264)]. 
See ```View Synthesis```[[local](#view-synthesis)].

#### Voxel

Text2Voxel v.1.0
[[colab](https://colab.research.google.com/github/tg-bomze/collection-of-notebooks/blob/master/Text2Voxel.ipynb)].

#### Cards

Using GANs to Create Fantastical Creatures
[[blog](https://ai.googleblog.com/2020/11/using-gans-to-create-fantastical.html)]. 
Leveraging Machine Learning for Game Development
[[blog](https://ai.googleblog.com/2021/03/leveraging-machine-learning-for-game.html)].

##### MTG

mtgencode: Generating Magic cards using deep, recurrent neural networks
[[code](https://github.com/billzorn/mtgencode)]
[[blog](https://www.mtgsalvation.com/forums/magic-fundamentals/custom-card-creation/612057-generating-magic-cards-using-deep-recurrent-neural)].

The AI That Learned Magic (the Gathering)
[[blog](https://www.vice.com/en/article/bmjke3/the-ai-that-learned-magic-the-gathering)].

### Text

Dialogs / story.

Plug-and-Blend: A Framework for Controllable Story Generation with Blended Control Codes
[[code](https://github.com/xxbidiao/plug-and-blend)]
[[arxiv](https://arxiv.org/abs/2104.04039)].

neural-storyteller
[[code](https://github.com/ryankiros/neural-storyteller)]
[[blog](https://medium.com/@samim/generating-stories-about-images-d163ba41e4ed)] - 
recurrent neural network that generates little stories about images.

Machines Making Movies \| Ross Goodwin & Oscar Sharp \| TEDxBoston
[[video](https://www.youtube.com/watch?v=uPXPQK83Z_Y)] - 
not movies itself, but a script written by a program created by 
[Ross Goodwin](https://rossgoodwin.com).

NeuralSnap
[[code](https://github.com/rossgoodwin/neuralsnap)] - 
Generates poetry from images using convolutional and recurrent neural networks.

See ```Language Models``` [[local](#language-models)].

### Audio

#### Music

##### Music Source Separation

(Facebook Research) Demucs Music Source Separation
[[code](https://github.com/facebookresearch/demucs)]
[[colab](https://colab.research.google.com/drive/1dC9nVxk3V_VPjUADsnFu8EiT-xnU1tGH)]
[[arxiv](https://arxiv.org/abs/2111.03600)]
[[arxiv](https://arxiv.org/abs/1911.13254)]
[[arxiv](https://arxiv.org/abs/1909.01174)].

KUIELab-MDX-Net: A Two-Stream Neural Network for Music Demixing
[[code](https://github.com/kuielab/mdx-net)]
[[arxiv](https://arxiv.org/abs/2111.12203)].

(Sony) D3Net: Densely connected multidilated DenseNet for music source separation
[[code](https://github.com/sony/ai-research-code/tree/master/d3net/music-source-separation)]
[[colab](https://colab.research.google.com/github/sony/ai-research-code/blob/master/d3net/music-source-separation/D3Net-MSS.ipynb)]
[[arxiv](https://arxiv.org/abs/2010.01733)].

A PyTorch implementation of DNN-based source separation
[[code](https://github.com/tky823/DNN-based_source_separation)].

Wave-U-Net: A Multi-Scale Neural Network for End-to-End Audio Source Separation
[[code](https://github.com/f90/Wave-U-Net)]
[[arxiv](https://arxiv.org/abs/1806.03185)].

Multi-scale Multi-band DenseNets for Audio Source Separation
[[code](https://github.com/tsurumeso/vocal-remover)]
[[code](https://github.com/Anjok07/ultimatevocalremovergui)]
[[arxiv](https://arxiv.org/abs/1706.09588)].

SigSep: Open Resources for Music Source Separation
[[project](https://sigsep.github.io)].

Current Trends in Audio Source Separation
[[video](https://www.youtube.com/watch?v=AB-F2JmI9U4)].

##### Music Generation

*TODO*: add a lot of existing projects and papers.

<img src="https://upload.wikimedia.org/wikipedia/commons/d/df/Container_01_KMJ.jpg" alt="container" title="Contains references to multiple projects" width="30" height="30"/>
Deep-Music
[[project](https://kasooja.github.io/2017/03/11/deep-music)] - 
Kartik Asooja page. 
*TODO*: add papers from here.

ming Visually Guided Sound Generation (BMVC 2021)
[[project](https://iashin.ai/SpecVQGAN)]
[[code](https://github.com/v-iashin/SpecVQGAN)]
[[colab](https://colab.research.google.com/drive/1pxTIMweAKApJZ3ZFqyBee3HtMqFpnwQ0?usp=sharing)]
[[video](https://www.youtube.com/watch?v=Bucb3nAa398)].

Francesco Marchetti- Convolutional Generative Adversarial Network for Scottish Music Generation
[[project](https://link.springer.com/chapter/10.1007/978-3-030-72914-1_13)]
[[video](https://www.youtube.com/watch?v=HJPvhT8jIjw)].

Deep Learning Techniques for Music Generation -- A Survey
[[code (part)](https://github.com/napulen/MiniBach)]
[[arxiv](https://arxiv.org/abs/1709.01620)].

JazzML: Computational Jazz Improvisation
[[code](https://github.com/evancchow/jazzml)]
[[paper](https://direct.mit.edu/comj/article-abstract/34/3/56/94318/Machine-Learning-of-Jazz-Grammars)].

MidiNet: A Convolutional Generative Adversarial Network for Symbolic-domain Music Generation
[[code](https://github.com/RichardYang40148/MidiNet)]
[[code (PyTorch)](https://github.com/annahung31/MidiNet-by-pytorch)]
[[data](https://github.com/wayne391/symbolic-music-datasets)]
[[arxiv](https://arxiv.org/abs/1703.10847)].

WaveNet: A Generative Model for Raw Audio
[[code (TensorFlow, unafficial)](https://github.com/ibab/tensorflow-wavenet)]
[[code (Keras, unafficial)](https://github.com/basveeling/wavenet)]
[[code (unafficial)](https://github.com/vincentherrmann/pytorch-wavenet)]
[[arxiv](https://arxiv.org/abs/1609.03499)]
[[blog](https://deepmind.com/blog/article/wavenet-generative-model-raw-audio)].

##### Music Stryle Transfer

ToneNet : A Musical Style Transfer
[[blog](https://towardsdatascience.com/tonenet-a-musical-style-transfer-c0a18903c910)].

#### Voice

Developing and Running Neural Audio in Constrained Environments
[[project](https://www.gdcvault.com/play/1026619/Machine-Learning-Summit-Developing-and)].

##### TTS

<img src="https://upload.wikimedia.org/wikipedia/commons/d/df/Container_01_KMJ.jpg" alt="container" title="Contains references to multiple projects" width="30" height="30"/>
Good collection of papers
[[project](https://github.com/coqui-ai/TTS-papers)].

Transfer Learning from Speaker Verification to Multispeaker Text-To-Speech Synthesis
[[code](https://github.com/CorentinJ/Real-Time-Voice-Cloning)].

Mozilla TTS: Text-to-Speech for all
[[code](https://github.com/mozilla/TTS)].

NVIDIA NeMo
[[code](https://github.com/NVIDIA/NeMo)] - a toolkit for conversational AI.

coqui-ai TTS
[[code](https://github.com/coqui-ai/TTS)].

Tacotron(1&2)
[[project](https://google.github.io/tacotron/index.html)]
[[project](https://pytorch.org/hub/nvidia_deeplearningexamples_tacotron2)]
[[code](https://github.com/NVIDIA/tacotron2)]
[[code](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/SpeechSynthesis/Tacotron2)]
[[colab](https://colab.research.google.com/github/pytorch/pytorch.github.io/blob/master/assets/hub/nvidia_deeplearningexamples_tacotron2.ipynb)]
[[arxiv](https://arxiv.org/abs/1703.10135v2)]
[[blog](https://ai.googleblog.com/2017/12/tacotron-2-generating-human-like-speech.html)].

PaddleSpeech (english, cheenese)
[[code](https://github.com/PaddlePaddle/PaddleSpeech)] - 
toolkit on ```PaddlePaddle```[[code](https://github.com/PaddlePaddle/Paddle)]
platform for a variety of critical tasks in speech and audio.

###### Soft

There are a lot of software/services for voice and tts. Some of them are below.

Replica Studios
[[soft](https://replicastudios.com)]
[[video](https://www.youtube.com/watch?v=jFxGmRPTEbo)] - AI voice actors for games, film & the metaverse.

murf.ai 
[[soft](https://murf.ai)].

play.ht
[[soft](https://play.ht)].

lovo.ai
[[soft](https://www.lovo.ai)].

See ```Synthesia```[[local](#Synthesia)].

#### Game Commenting

AI generating real-time football commentary (2019)
[[code](https://github.com/ChintanTrivedi/football_ai_commentary)]
[[video](https://www.youtube.com/watch?v=p9AmkiG8UeI)].

#### Cources

This channel in total [[video](https://www.youtube.com/channel/UCZPFjMe1uRSirmSpznqvJfQ)] and the cources below.

Deep Learning (for Audio) with Python course
[[video](https://www.youtube.com/watch?v=fMqL5vckiU0&list=PL-wATfeyAMNrtbkCNsLcpoAyBBRJZVlnf)]

Audio Signal Processing for Machine Learning course
[[video](https://www.youtube.com/watch?v=iCwMQJnKk2c&list=PL-wATfeyAMNqIee7cH3q1bh4QJFAaeNv0)]

Sound Generation with Neural Networks course
[[video](https://www.youtube.com/watch?v=Ey8IZQl_lKs&list=PL-wATfeyAMNpEyENTc-tVH5tfLGKtSWPp)]

PyTorch for Audio + Music Processing course
[[video](https://www.youtube.com/watch?v=gp2wZqDoJ1Y&list=PL-wATfeyAMNoirN4idjev6aRu8ISZYVWm)]

### Video: trailers/cinematics

World-Consistent Video-to-Video Synthesis (ECCV 2020)
[[project](https://nvlabs.github.io/wc-vid2vid)]
[[code](https://github.com/NVlabs/imaginaire/blob/master/projects/wc_vid2vid/README.md)]
[[data](https://www.cityscapes-dataset.com)]
[[paper](https://nvlabs.github.io/wc-vid2vid/files/wc-vid2vid.pdf)]
[[arxiv](https://arxiv.org/abs/2007.08509)]
[[video](https://www.youtube.com/watch?v=b2P39sS2kKo)] - 
part of ```Imaginaire```[[local](#Imaginaire)].

Scale-aware Insertion of Virtual Objects in Monocular Videos (ISMAR 2020)
[[data](https://metric-tree.github.io)]
[[arxiv](https://arxiv.org/abs/2012.02371)].

A Comprehensive Tutorial on Video Modeling (CVPR 2020)
[[project](https://bryanyzhu.github.io/videomodeling.github.io)].

Few-shot Video-to-Video Synthesis (NeurIPS 2019)
[[project](https://nvlabs.github.io/few-shot-vid2vid)]
[[code](https://github.com/NVlabs/imaginaire/blob/master/projects/fs_vid2vid/README.md)]
[[code (Previous Implementation)](https://github.com/NVlabs/few-shot-vid2vid)]
[[data](https://niessnerlab.org/projects/roessler2018faceforensics.html)]
[[paper](https://nvlabs.github.io/few-shot-vid2vid/main.pdf)]
[[arxiv](https://arxiv.org/abs/1910.12713)]
[[video](https://www.youtube.com/watch?v=8AZBuyEuDqc)] - 
part of ```Imaginaire```[[local](#Imaginaire)].

Compositional Video Prediction (ICCV 2019)
[[project](https://judyye.github.io/CVP)]
[[code](https://github.com/JudyYe/CVP)]
[[arxiv](https://arxiv.org/abs/1908.08522)].

Video-to-Video Synthesis (NeurIPS 2018)
[[project](https://tcwang0509.github.io/vid2vid)]
[[code](https://github.com/NVlabs/imaginaire/blob/master/projects/vid2vid/README.md)]
[[code (Previous Implementation)](https://github.com/NVIDIA/vid2vid)]
[[data](https://www.cityscapes-dataset.com)]
[[paper](https://tcwang0509.github.io/vid2vid/paper_vid2vid.pdf)]
[[arxiv](https://arxiv.org/abs/1808.06601)]
[[video](https://www.youtube.com/watch?v=5zlcXTCpQqM)]
[[video](https://www.youtube.com/watch?v=GrP_aOSXt5U)] - 
part of ```Imaginaire```[[local](#Imaginaire)].

See ```Radiance Fields```[[local](#radiance-fields)].

#### Face Video

One-Shot Free-View Neural Talking-Head Synthesis for Video Conferencing (CVPR 2021)
[[project](https://nvlabs.github.io/face-vid2vid)]
[[code (unafficial)](https://github.com/zhanglonghao1992/One-Shot_Free-View_Neural_Talking_Head_Synthesis)]
[[demo](http://imaginaire.cc/vid2vid-cameo)]
[[paper](https://nvlabs.github.io/face-vid2vid/main.pdf)]
[[arxiv](https://arxiv.org/abs/2011.15126)]
[[video](https://www.youtube.com/watch?v=nLYg9Waw72U)]
[[video](https://www.youtube.com/watch?v=smrcnZ5Eg4A)].

See ```Face Image```[[local](#face-image)].

## Code

### Code Generation

OpenAI Codex
[[code](https://github.com/openai/human-eval)]
[[arxiv](https://arxiv.org/abs/2107.03374)]
[[blog](https://openai.com/blog/openai-codex)]
[[video](https://www.youtube.com/watch?v=1hJdBNYTNmQ)] - 
model that powers ```GitHub Copilot```[[soft](https://copilot.github.com)].

See ```Language Models```[[local](#language-models)].

### Finding Bugs

#### Ubisof

CLEVER: Combining Code Metrics with Clone Detection for Just-In-Time Fault Prevention and Resolution in Large Industrial Projects
[[project](https://montreal.ubisoft.com/en/clever-combining-code-metrics-with-clone-detection-for-just-in-time-fault-prevention-and-resolution-in-large-industrial-projects-2)]
[[project](https://montreal.ubisoft.com/en/ubisoft-la-forge-presents-the-commit-assistant)]
[[paper](https://static-wordpress.akamaized.net/montreal.ubisoft.com/wp-content/uploads/2018/05/03173315/ICSE-CE-MSR-165.pdf)]
[[paper](https://montreal.ubisoft.com/wp-content/uploads/2018/03/clever-commit-msr18.pdf)]
[[blog](https://news.ubisoft.com/en-us/article/4tdnCF5t0JOxTMYLVfnvvf/ubisoft-and-mozilla-partner-to-develop-ai-coding-tools)]
[[blog](https://blog.mozilla.org/futurereleases/2019/02/12/making-the-building-of-firefox-faster-for-you-with-clever-commit-from-ubisoft)]
[[video](https://www.youtube.com/watch?v=I5C4FUvDyCc)].

Better C++ using Machine Learning on Large Projects
[[supplement](https://github.com/CppCon/CppCon2018/blob/master/Presentations/better_cpp_using_machine_learning_on_large_projects/better_cpp_using_machine_learning_on_large_projects__nicolas_fleury_mathieu_nayrolles__cppcon_2018.pdf)]
[[video](https://www.youtube.com/watch?v=QDvic0QNtOY)].

## Testing

Deep Reinforcement Learning for Game Testing at EA with Konrad Tollmar
[[blog](https://twimlai.com/deep-reinforcement-learning-for-game-testing-at-ea-with-konrad-tollmar)]
[[video](https://www.youtube.com/watch?v=uCxAvGN9F6E)].

GTC 2021: Towards Advanced Game Testing With AI
[[blog](https://www.ea.com/seed/news/gtc-2021-towards-advanced-game-testing-with-ai)]
[[video](https://www.youtube.com/watch?v=E8N01hsatFg)].

### Bots Playing Game

Improving Playtesting Coverage via Curiosity Driven Reinforcement Learning Agents (CoG 2021)
[[arxiv](https://arxiv.org/abs/2103.13798)]
[[blog](https://www.ea.com/seed/news/cog2021-curiosity-driven-rl-agents)]
[[video](https://www.youtube.com/watch?v=UmoJQT-gEM8)]
[[video](https://video.itu.dk/video/71684715/improving-playtesting-coverage-via)].

Augmenting Automated Game Testing with Deep Reinforcement Learning (CoG 2020)
[[paper](https://media.contentapi.ea.com/content/dam/ea/seed/presentations/seed-augmenting-automated-game-testing-with-deep-reinforcement-learning.pdf)]
[[arxiv](https://arxiv.org/abs/2103.15819)]
[[supplement](https://media.contentapi.ea.com/content/dam/ea/seed/presentations/seed-rework2021-linusgisslen-rl-game-testing.pdf)]
[[blog](https://www.ea.com/seed/news/automated-game-testing-deep-reinforcement-learning)]
[[video](https://www.youtube.com/watch?v=2n8Tjz0S2rs)]
[[video](https://www.youtube.com/watch?v=NkhrqiOVA64)].

Successfully Use Deep Reinforcement Learning in Testing and NPC Development
[[project](https://www.gdcvault.com/play/1026732/Machine-Learning-Summit-Successfully-Use)]
[[video](https://www.youtube.com/watch?v=Q5RAE73zCKQ)].

OpenAI Baselines
[[code](https://github.com/openai/baselines)] - 
high-quality implementations of reinforcement learning algorithms.

OpenAI Emergent Tool Use From Multi-Agent Autocurricula
[[arxiv](https://arxiv.org/abs/1909.07528)]
[[blog](https://openai.com/blog/emergent-tool-use)].

OpenAI Emergent Complexity via Multi-Agent Competition
[[arxiv](https://arxiv.org/abs/1710.03748)]
[[blog](https://openai.com/blog/competitive-self-play)]
[[video](https://www.youtube.com/watch?v=OBcjhp4KSgQ)].

OpenAI Proximal Policy Optimization Algorithms
[[arxiv](https://arxiv.org/abs/1707.06347)]
[[blog](https://openai.com/blog/openai-baselines-ppo)]

Control Strategies for Physically Simulated Characters Performing Two-player Competitive Sports
\[[paper][Control Strategies for Physically Simulated Characters Performing Two-player Competitive Sports]\]
[[blog](https://ai.facebook.com/research/publications/control-strategies-for-physically-simulated-characters-performing-two-player-competitive-sports)].

Artificial intelligence through learning or Pavlovian algorithm?
[[blog](https://montreal.ubisoft.com/en/artificial-intelligence-through-learning-or-pavlovian-algorithm)].

Aslo see ```Environments```[[local](#environments)].

See ```DeepMind```[[local](#deepmind)].

### Graphics

Back to the Feature: Learning Robust Camera Localization from Pixels to Pose (CVPR 2021)
[[project](https://psarlin.com/pixloc)]
[[code](https://github.com/cvg/pixloc)]
[[arxiv](https://arxiv.org/abs/2103.09213)]
[[slides](https://psarlin.com/pixloc/assets/pixloc_slides.pdf)]
[[video](https://www.youtube.com/watch?v=vPkXhKQn2oI)] - 
possible usage is to test scene rendering with comparision to reference images.

Graphical Glitch Detection in Video Games Using CNNs
[[paper](https://media.contentapi.ea.com/content/dam/ea/seed/presentations/garcialing2020-graphical-glitch-detection-in-video-games-using-cnns.pdf)]
[[blog](https://www.ea.com/seed/news/graphical-glitch-detection-convolutional-neural-networks)]
[[video](https://www.youtube.com/watch?v=mkPwlrm7LR0)].

# Game analytics

Not sure if it should be covered here. Analytics can be used for many areas of game development 
(example: ```prevent-cheating/fraud``` [[local](#prevent-cheating/fraud)]), 
but are there any gamedev-specific application of ML in analytics?

# Common Models/Papers/Repos

Models/Papers/Repos referenced in multiple places.

## DeepMind

DeepMind Research
[[code](https://github.com/deepmind/deepmind-research)] - 
contains implementations and illustrative code to accompany DeepMind publications.

### EfficientZero

Mastering Atari Games with Limited Data (NeurIPS 2021)
[[code](https://github.com/YeWR/EfficientZero)]
[[arxiv](https://arxiv.org/abs/2111.00210)].

### MuZero

MuZero: Mastering Go, chess, shogi and Atari without rules
[[blog](https://deepmind.com/blog/article/muzero-mastering-go-chess-shogi-and-atari-without-rules)].

MuZero General
[[code](https://github.com/werner-duvaud/muzero-general)] - 
a commented and documented implementation of MuZero based on the Google DeepMind paper (Nov 2019) and the associated pseudocode.

muzero-pytorch
[[code](https://github.com/koulanurag/muzero-pytorch)].

MuZero Vs. AlphaZero in Tensorflow
[[code](https://github.com/kaesve/muzero)] - 
readable, commented, well documented, and conceptually easy implementation of the AlphaZero and MuZero algorithms 
based on the popular AlphaZero-General implementation.

### AplhaZero

AlphaZero: Shedding new light on chess, shogi, and Go
[[blog](https://deepmind.com/blog/article/alphazero-shedding-new-light-grand-games-chess-shogi-and-go)].

Alpha Zero General
[[code](https://github.com/suragnair/alpha-zero-general)] - 
a simplified, highly flexible, commented and (hopefully) easy to understand implementation of self-play 
based reinforcement learning based on the AlphaGo Zero paper.

### AlphaGo

AlphaGo the story so far
[[blog](https://deepmind.com/research/case-studies/alphago-the-story-so-far)].

### DeepMind Environments

DeepMind Lab
[[code](https://github.com/deepmind/lab)].

PySC2
[[code](https://github.com/deepmind/pysc2)] - 
StarCraft II Learning Environment.

## Language Models

GPT Neo
[[project](https://www.eleuther.ai/projects/gpt-neo)]
[[code](https://github.com/EleutherAI/gpt-neo)] - 
an implementation of model & data parallel GPT3-like models using the mesh-tensorflow library.

GPT-NeoX
[[project](https://www.eleuther.ai/projects/gpt-neox)]
[[code](https://github.com/EleutherAI/gpt-neox)].

GPT-J-6B
[[code](https://github.com/kingoflolz/mesh-transformer-jax/#gpt-j-6b)]
[[colab](https://colab.research.google.com/github/kingoflolz/mesh-transformer-jax/blob/master/colab_demo.ipynb)]
[[blog](https://arankomatsuzaki.wordpress.com/2021/06/04/gpt-j)] - 
a 6 billion parameter, autoregressive text generation model trained on 
[The Pile](https://pile.eleuther.ai). Demo is [here](https://6b.eleuther.ai).

DialoGPT
[[project](https://www.microsoft.com/en-us/research/project/large-scale-pretraining-for-response-generation)]
[[code](https://github.com/microsoft/DialoGPT)]
[[paper](https://www.microsoft.com/en-us/research/uploads/prod/2019/11/1911.00536.pdf)]
[[arxiv](https://arxiv.org/abs/1911.00536)] - 
Large-scale Pretrained Response Generation Model.

NVIDIA Megatron
[[code](https://github.com/NVIDIA/Megatron-LM)].

GPT-3 Powers the Next Generation of Apps
[[blog](https://openai.com/blog/gpt-3-apps)] - has some examples of GPT-3 usage.

GLaM: Efficient Scaling of Language Models with Mixture-of-Experts
[[arxiv](https://arxiv.org/abs/2112.06905)]
[[blog](https://ai.googleblog.com/2021/12/more-efficient-in-context-learning-with.html)].

Scaling Language Models: Methods, Analysis & Insights from Training Gopher
[[paper](https://storage.googleapis.com/deepmind-media/research/language-research/Training%20Gopher.pdf)]
[[arxiv](https://arxiv.org/abs/2112.11446)]
[[blog](https://deepmind.com/blog/article/language-modelling-at-scale)].

Google GLaM Vs DeepMind Gopher: Who Wins The Large Language Model Race
[[blog](https://analyticsindiamag.com/google-glam-vs-deepmind-gopher-who-wins-the-large-language-model-race)].

See ```Hugging Face Transformers```[[local](#hugging-face-transformers)].

## Hugging Face Transformers

<img src="https://upload.wikimedia.org/wikipedia/commons/d/df/Container_01_KMJ.jpg" alt="container" title="Contains references to multiple projects" width="30" height="30"/>
Transformers
[[project](https://huggingface.co/transformers)]
[[code](https://github.com/huggingface/transformers)] - 
Hugging Face Machine Learning models, also see 
Hugging Face Course [[course](https://huggingface.co/course)].

## Speech recognition

*TODO*: add projects.

<img src="https://upload.wikimedia.org/wikipedia/commons/d/df/Container_01_KMJ.jpg" alt="container" title="Contains references to multiple projects" width="30" height="30"/>
speech-to-text topic on github
[[code](https://github.com/topics/speech-to-text)] - 
a lot of good projects.

## Classification

MMClassification
[[code](https://github.com/open-mmlab/mmclassification)] - 
open source image classification toolbox based on PyTorch.
*TODO*: add pertinent models from MMClassification here explicitly.

<a name="MMFewshot"></a>
MMFewshot
[[code](https://github.com/open-mmlab/mmfewshot)] - 
open source few shot learning toolbox based on PyTorch.
*TODO*: add pertinent models from MMFewshot here explicitly.

Common Objects in 3D: Large-Scale Learning and Evaluation of Real-life 3D Category Reconstruction (ICCV 2021)
[[project](https://henzler.github.io/publication/common_3d_objects)]
[[code](https://github.com/facebookresearch/co3d)]
[[data](https://ai.facebook.com/datasets/co3d-downloads)]
[[arxiv](https://arxiv.org/abs/2109.00512)]
[[video](https://www.youtube.com/watch?v=hMx9nzG50xQ)].

## Detection and segmentation

MMDetection
[[code](https://github.com/open-mmlab/mmdetection)] - 
open source object detection toolbox based on PyTorch.
*TODO*: add pertinent models from MMDetection here explicitly.

MMDetection3D
[[code](https://github.com/open-mmlab/mmdetection3d)] - 
open source object detection toolbox based on PyTorch.
*TODO*: add pertinent models from MMDetection3D here explicitly.

See ```MMFewshot```[[local](#MMFewshot)].

### Detectron

Detectron2 is Facebook AI Research's next generation library that provides 
state-of-the-art detection and segmentation algorithms [[code](https://github.com/facebookresearch/detectron2)].

Projects that are built on detectron2 [[code](https://github.com/facebookresearch/detectron2/tree/main/projects)].

Previous version of Detectron [[code](https://github.com/facebookresearch/Detectron)].

## Tracking

MMTracking
[[code](https://github.com/open-mmlab/mmtracking)] - 
open source video perception toolbox based on PyTorch.
*TODO*: add pertinent models from MMTracking here explicitly.

## Depth Estimation

Learning High Fidelity Depths of Dressed Humans by Watching Social Media Dance Videos (CVPR 2021)
[[project](https://www.yasamin.page/hdnet_tiktok)]
[[code](https://github.com/yasaminjafarian/HDNet_TikTok)]
[[data](https://www.yasamin.page/hdnet_tiktok#h.jr9ifesshn7v)]
[[paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Jafarian_Learning_High_Fidelity_Depths_of_Dressed_Humans_by_Watching_Social_CVPR_2021_paper.pdf)]
[[supplement](https://openaccess.thecvf.com/content/CVPR2021/supplemental/Jafarian_Learning_High_Fidelity_CVPR_2021_supplemental.zip)]
[[arxiv](https://arxiv.org/abs/2103.03319)]
[[video](https://www.youtube.com/watch?v=EFJ8WXdKghs)]
[[video](https://www.youtube.com/watch?v=VYArtX_Ng_U)]. 
See ```Clothes```[[local](#clothes)].

Pano3D: A Holistic Benchmark and a Solid Baseline for 360o Depth Estimation (CVPR 2021)
[[project](https://vcl3d.github.io/Pano3D)]
[[code](https://github.com/VCL3D/Pano3D)]
[[data](https://vcl3d.github.io/Pano3D/download)]
[[arxiv](https://arxiv.org/abs/2109.02749)]. 
*TODO*: see papers references on project site.

3D Imaging with an RGB Camera and a single SPAD Transient (ECCV 2020)
[[project](https://www.computationalimaging.org/publications/single_spad)]
[[code](https://github.com/computational-imaging/single_spad_depth)]
[[paper](https://www.computationalimaging.org/wp-content/uploads/2020/07/eccv2020.pdf)]
[[supplement](https://drive.google.com/file/d/1O7LSTxbJW-AhgbgKeYwH5fVb0_eeqT00/view)]
[[video](https://www.youtube.com/watch?v=j91H56iqxJs)].

Spherical View Synthesis for Self-Supervised 360 Depth Estimation (3DV 2019)
[[project](https://vcl3d.github.io/SphericalViewSynthesis)]
[[code](https://github.com/VCL3D/SphericalViewSynthesis)]
[[data](https://vcl3d.github.io/3D60)]
[[models](https://github.com/VCL3D/SphericalViewSynthesis/releases)]
[[arxiv](https://arxiv.org/abs/1909.08112)]
[[video](https://www.youtube.com/watch?v=7sWUyoJNe-U)]. 
See ```View Synthesis```[[local](#view-synthesis)].

Unsupervised Monocular Depth Estimation with Left-Right Consistency (CVPR 2017)
[[project](https://visual.cs.ucl.ac.uk/pubs/monoDepth)]
[[code](https://github.com/mrharicot/monodepth)]
[[models](http://visual.cs.ucl.ac.uk/pubs/monoDepth/models)]
[[results](http://visual.cs.ucl.ac.uk/pubs/monoDepth/results)]
[[arxiv](https://arxiv.org/abs/1609.03677)]
[[video](https://www.youtube.com/watch?v=go3H2gU-Zck)]
[[video](https://www.youtube.com/watch?v=jI1Qf7zMeIs)]
[[video](https://www.youtube.com/watch?v=v8cpDQ22bSg)].

## Action

Where2Act: From Pixels to Actions for Articulated 3D Objects (ICCV 2021)
[[project](https://cs.stanford.edu/~kaichun/where2act)]
[[code](https://github.com/daerduoCarey/where2act)]
[[arxiv](https://arxiv.org/abs/2101.02692)]
[[slides](https://cs.stanford.edu/~kaichun/where2act/slides.pdf)]
[[poster](https://cs.stanford.edu/~kaichun/where2act/poster.pdf)]
[[video](https://www.youtube.com/watch?v=cdMSZru3Aa8)].

## Human

### Datasets

HUMBI: A Large Multiview Dataset of Human Body Expressions and Benchmark Challenge (CVPR 2020)
[[project](https://humbi-data.net)]
[[code](https://github.com/zhixuany/HUMBI)]
[[paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Yu_HUMBI_A_Large_Multiview_Dataset_of_Human_Body_Expressions_CVPR_2020_paper.pdf)]
[[arxiv](https://arxiv.org/abs/2110.00119)]
[[supplement](https://openaccess.thecvf.com/content_CVPR_2020/supplemental/Yu_HUMBI_A_Large_CVPR_2020_supplemental.zip)]
[[video](https://www.youtube.com/watch?v=Vd1VEmfM3YQ)]
[[video](https://www.youtube.com/watch?v=bHc0CmXRUO4)]
[[video](https://www.youtube.com/watch?v=1iVjFgYq8ZU)].

### Action

MMAction2
[[code](https://github.com/open-mmlab/mmaction2)] - 
open-source toolbox for video understanding based on PyTorch.
Not only, but mostly for human, so for now it is here.
*TODO*: add pertinent models from MMAction2 here explicitly.

### Clothes

<img src="https://upload.wikimedia.org/wikipedia/commons/d/df/Container_01_KMJ.jpg" alt="container" title="Contains references to multiple projects" width="30" height="30"/>
Clothes-3D
[[project](https://github.com/lzhbrian/Clothes-3D)]. 
*TODO*: add papers from here.

<img src="https://upload.wikimedia.org/wikipedia/commons/d/df/Container_01_KMJ.jpg" alt="container" title="Contains references to multiple projects" width="30" height="30"/>
awesome clothed people
[[project](https://github.com/weihaox/awesome-clothed-human)].
*TODO*: add papers from here.

Point-Based Modeling of Human Clothing (ICCV 2021)
[[project](https://saic-violet.github.io/point-based-clothing)]
[[code](https://github.com/saic-vul/point_based_clothing)]
[[data](https://chalearnlap.cvc.uab.cat/dataset/38/description)]
[[paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Zakharkin_Point-Based_Modeling_of_Human_Clothing_ICCV_2021_paper.pdf)]
[[arxiv](https://arxiv.org/abs/2104.08230)]
[[video](https://www.youtube.com/watch?v=kFrAu415kDU)].

The Power of Points for Modeling Humans in Clothing (ICCV 2021)
[[project](https://qianlim.github.io/POP)]
[[code](https://github.com/qianlim/POP)]
[[data](https://pop.is.tue.mpg.de)]
[[paper](https://ps.is.mpg.de/uploads_file/attachment/attachment/655/POP_camera_ready.pdf)]
[[arxiv](https://arxiv.org/abs/2109.01137)]
[[supplement](https://ps.is.mpg.de/uploads_file/attachment/attachment/656/POP_supp.pdf)]
[[video](https://www.youtube.com/watch?v=JY5OI74yJ4w)].

SCALE: Modeling Clothed Humans with a Surface Codec of Articulated Local Elements (CVPR 2021)
[[project](https://qianlim.github.io/SCALE)]
[[code](https://github.com/qianlim/SCALE)]
[[arxiv](https://arxiv.org/abs/2104.07660)]
[[poster](https://ps.is.mpg.de/uploads_file/attachment/attachment/650/SCALE_poster_CVPR_final_compressed.pdf)]
[[video](https://www.youtube.com/watch?v=-EvWqFCUb7U)].

SCANimate: Weakly Supervised Learning of Skinned Clothed Avatar Networks (CVPR 2021)
[[project](https://scanimate.is.tue.mpg.de)]
[[code](https://github.com/shunsukesaito/SCANimate)]
[[demo](https://scanimate.is.tue.mpg.de/#animations)]
[[paper](https://scanimate.is.tue.mpg.de/media/upload/paper/SCANimate.pdf)]
[[supplement](https://scanimate.is.tue.mpg.de/media/upload/poster/CVPR_poster_SCANimate.pdf)]
[[supplement](https://scanimate.is.tue.mpg.de/media/upload/paper/SCANimate-supp.pdf)]
[[video](https://www.youtube.com/watch?v=ohavL55Oznw)]
[[video](https://www.youtube.com/watch?v=EeNFvmNuuog)].

Neural 3D Clothes Retargeting from a Single Image
[[arxiv](https://arxiv.org/abs/2102.00062)].

CAPE: Clothed Auto Person Encoding, 
Learning to Dress 3D People in Generative Clothing (CVPR 2020)
[[project](https://cape.is.tue.mpg.de)]
[[code](https://github.com/QianliM/CAPE)]
[[data](https://cape.is.tue.mpg.de/dataset.html)]
[[paper](https://cape.is.tuebingen.mpg.de/media/upload/CAPE_paper.pdf)]
[[paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Ma_Learning_to_Dress_3D_People_in_Generative_Clothing_CVPR_2020_paper.pdf)]
[[arxiv](https://arxiv.org/abs/1907.13615)]
[[supplement](https://cape.is.tuebingen.mpg.de/media/upload/CAPE_suppmat.pdf)]
[[slides](https://cape.is.tuebingen.mpg.de/media/upload/CAPE_slides.pdf)].
[[video](https://www.youtube.com/watch?v=e4W-hPFNwDE)].

TailorNet: Predicting Clothing in 3D as a Function of Human Pose, Shape and Garment Style (CVPR 2020)
[[project](https://virtualhumans.mpi-inf.mpg.de/tailornet)]
[[code](https://github.com/chaitanya100100/TailorNet)]
[[data](https://github.com/zycliao/TailorNet_dataset)]
[[paper](https://virtualhumans.mpi-inf.mpg.de/tailornet/patel20tailornet.pdf)]
[[arxiv](https://arxiv.org/abs/2003.04583)]
[[supplement](https://virtualhumans.mpi-inf.mpg.de/tailornet/patel20tailornet_supp.pdf)]
[[video](https://www.youtube.com/watch?v=F0O21a_fsBQ)]
[[video](https://www.youtube.com/watch?v=vg7a52zObjs)].

ClothCap: Seamless 4D Clothing Capture and Retargeting (SIGGRAPH 2017)
[[project](https://clothcap.is.tue.mpg.de)]
[[project](https://ps.is.mpg.de/publications/pons-moll-siggraph2017)]
[[paper](https://ps.is.mpg.de/uploads_file/attachment/attachment/374/clothcap.pdf)]
[[paper](https://virtualhumans.mpi-inf.mpg.de/papers/ponsmollSIGGRAPH17clothcap/ponsmollSIGGRAPH17clothcap.pdf)]
[[video](https://www.youtube.com/watch?v=dVxj8tzx04U)]

#### Virtual Try-on (VTON)

<img src="https://upload.wikimedia.org/wikipedia/commons/d/df/Container_01_KMJ.jpg" alt="container" title="Contains references to multiple projects" width="30" height="30"/>
A Curated List of Awesome Virtual Try-on (VTON) Research
[[project](https://github.com/minar09/awesome-virtual-try-on)].
*TODO*: add papers from here.

### Person Pose and Shape

*TODO*: review structure of inner topics.

*TODO*: are all papers works only with images, or may be wirth video (and use temporal information), or 3D scans? Should I structure?

<img src="https://upload.wikimedia.org/wikipedia/commons/d/df/Container_01_KMJ.jpg" alt="container" title="Contains references to multiple projects" width="30" height="30"/>
Awesome 3D Body Papers
[[project](https://github.com/3DFaceBody/awesome-3dbody-papers)].
*TODO*: add papers from here.

MMHuman
[[code](https://github.com/open-mmlab/mmhuman3d)] - 
open source PyTorch-based codebase for the use of 3D human parametric models in computer vision and computer graphics.
*TODO*: add pertinent models from MMHuman here explicitly.

OpenPose
[[code](https://github.com/CMU-Perceptual-Computing-Lab/openpose)] - 
mostly 2D.

Meysam Madadi: deep learning advances on human pose and shape estimation (2020)
[[video](https://www.youtube.com/watch?v=bnvuOOgQ2zY)].

See ```Statistical Body Models```[[local](#statistical-body-models).

#### Person Pose Detection

BodyPoseNet (2021)
[[blog](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/bodyposenet)] - 
NVidia's BodyPoseNet model based on [Train Adapt Optimize (TAO) Toolkit](https://developer.nvidia.com/tao-toolkit).

#### Person Segmentation

#### Person Part Segmentation

#### Person Pose Estimation

MMPose
[[code](https://github.com/open-mmlab/mmpose)] - 
open-source toolbox for pose estimation based on PyTorch.
*TODO*: add pertinent models from MMPose here explicitly.

Real-time 3D Multi-person Pose Estimation Demo
[[code](https://github.com/Daniil-Osokin/lightweight-human-pose-estimation-3d-demo.pytorch)].

XNect: Real-time Multi-Person 3D Motion Capture with a Single RGB Camera (SIGGRAPH 2020)
[[project](https://vcai.mpi-inf.mpg.de/projects/XNect)]
[[code (partial)](https://github.com/mehtadushy/SelecSLS-Pytorch)]
[[paper](https://vcai.mpi-inf.mpg.de/projects/XNect/content/XNect_SIGGRAPH2020.pdf)]
[[arxiv](https://arxiv.org/abs/1907.00837)]
[[supplement](https://vcai.mpi-inf.mpg.de/projects/XNect/content/XNect_supp_SIGGRAPH2020.pdf)].

Epipolar Transformers (CVPR 2020)
[[code](https://github.com/yihui-he/epipolar-transformers)]
[[arxiv](https://arxiv.org/abs/2005.04551)]
[[video](https://www.youtube.com/playlist?list=PLkz610aVEqV-f4Ws0pH0e8Nm_2wTGI1yP)].

Keep it SMPL: Automatic Estimation of 3D Human Pose and Shape from a Single Image (ECCV 2016)
[[project](https://smplify.is.tuebingen.mpg.de)]
[[paper](https://files.is.tue.mpg.de/black/papers/BogoECCV2016.pdf)]
[[video](https://www.youtube.com/watch?v=eUnZ2rjxGaE)]
[[video](https://www.youtube.com/watch?v=OgX49T2Cqdo)].

#### Hand Pose Estimation

<img src="https://upload.wikimedia.org/wikipedia/commons/d/df/Container_01_KMJ.jpg" alt="container" title="Contains references to multiple projects" width="30" height="30"/>
Awesome Hand Pose Estimation
[[project](https://github.com/xinghaochen/awesome-hand-pose-estimation)].
*TODO*: add papers from here.

HuMoR: 3D Human Motion Model for Robust Pose Estimation (ICCV 2021)
[[project](https://geometry.stanford.edu/projects/humor)]
[[code](https://github.com/davrempe/humor)]
[[paper](https://geometry.stanford.edu/projects/humor/docs/humor.pdf)]
[[arxiv](https://arxiv.org/abs/2105.04668)]
[[supplement](https://geometry.stanford.edu/projects/humor/supp.html)].

3D Hand Pose Estimation Using Convolutional Neural Networks (2018)
[[project](https://www.microsoft.com/en-us/research/video/3d-hand-pose-estimation-using-convolutional-neural-networks)]
[[video](https://www.youtube.com/watch?v=aE7kW4b6CjA)].

#### Head Pose Estimation

DeepHeadPose
[[code](https://github.com/DriverDistraction/DeepHeadPose)]
[[code](https://github.com/natanielruiz/deep-head-pose)]
[[arxiv](https://arxiv.org/abs/1710.00925)].

#### Person Shape Capture

Hierarchical Kinematic Probability Distributions for 3D Human Shape and Pose Estimation from Images in the Wild (ICCV 2021)
[[code](https://github.com/akashsengupta1997/hierarchicalprobabilistic3dhuman)]
[[arxiv](https://arxiv.org/abs/2110.00990)]
[[video](https://www.youtube.com/watch?v=qVrvOebDBs4)]
[[video](https://www.youtube.com/watch?v=w7k9UC3sfGA)].

TightCap: 3D Human Shape Capture with Clothing Tightness Field (TOG 2021)
[[project](https://chenxin.tech/files/Paper/TOG2021_TightCap/project_page_TightCap/index.htm)]
[[code](https://github.com/ChenFengYe/TightCap)]
[[paper](https://chenxin.tech/files/Paper/TOG2021_TightCap/project_page_TightCap/data/TightCap.pdf)]
[[arxiv](https://arxiv.org/abs/1904.02601)]
[[video](https://chenxin.tech/files/Paper/TOG2021_TightCap/project_page_TightCap/data/video.mp4)].

DeepCap: Monocular Human Performance Capture Using Weak Supervision
[[project](https://people.mpi-inf.mpg.de/~mhaberma/projects/2020-cvpr-deepcap)]
[[data](https://gvv-assets.mpi-inf.mpg.de)]
[[paper](https://people.mpi-inf.mpg.de/~mhaberma/projects/2020-cvpr-deepcap/data/paper.pdf)]
[[arxiv](https://arxiv.org/abs/2003.08325)]
[[slides](https://vision-and-learning-lab-ualberta.github.io/talk_slides/ji_july_19.pdf)]
[[supplement](https://people.mpi-inf.mpg.de/~mhaberma/projects/2020-cvpr-deepcap/data/supp.pdf)]
[[video](https://www.youtube.com/watch?v=C4eDrvJ9aBs)]
[[video](https://people.mpi-inf.mpg.de/~mhaberma/projects/2020-cvpr-deepcap/data/video.mp4)]
[[video](https://people.mpi-inf.mpg.de/~mhaberma/projects/2020-cvpr-deepcap/data/talk.mp4)]
[[video](https://people.mpi-inf.mpg.de/~mhaberma/projects/2020-cvpr-deepcap/data/teaser.mp4)]. 
*TODO*: see papers used the dataset below.

Implicit Functions in Feature Space for 3D Shape Reconstruction and Completion (CVPR 2020)
[[project](https://virtualhumans.mpi-inf.mpg.de/ifnets)]
[[code](https://github.com/jchibane/if-net)]
[[paper](https://virtualhumans.mpi-inf.mpg.de/papers/chibane20ifnet/chibane20ifnet.pdf)]
[[arxiv](https://arxiv.org/abs/2003.01456)]
[[video](https://www.youtube.com/watch?v=cko07jINRZg)].

PIFu: Pixel-Aligned Implicit Function for High-Resolution Clothed Human Digitization (ICCV 2019)
[[project](https://shunsukesaito.github.io/PIFu)]
[[code](https://github.com/shunsukesaito/PIFu)]
[[arxiv](https://arxiv.org/abs/1905.05172)]
[[video](https://www.youtube.com/watch?v=S1FpjwKqtPs)].

Learning to Reconstruct People in Clothing from a Single RGB Camera (CVPR 2019)
[[project](https://virtualhumans.mpi-inf.mpg.de/octopus)]
[[code](https://github.com/thmoa/octopus)]
[[paper](https://virtualhumans.mpi-inf.mpg.de/papers/alldieck19cvpr/alldieck19cvpr.pdf)]
[[video](https://www.youtube.com/watch?v=_wuEru4WeDw)].

#### Person Dense Pose

DensePose: Dense Human Pose Estimation In The Wild (ICCV 2019)
[[project](http://densepose.org)]
[[code](https://github.com/facebookresearch/DensePose)]
[[arxiv](https://arxiv.org/abs/1802.00434)]
[[video](https://www.youtube.com/watch?v=Dhkd_bAwwMc)].

#### Pose Retargeting

Skeleton-Aware Networks for Deep Motion Retargeting (SIGGRAPH 2020)
[[project](https://deepmotionediting.github.io/retargeting)]
[[code](https://github.com/DeepMotionEditing/deep-motion-editing)]
[[arxiv](https://arxiv.org/abs/2005.05732)]
[[video](https://www.youtube.com/watch?v=ym8Tnmiz5N8)].

See ```Learning Character-Agnostic Motion for Motion Retargeting in 2D```[[local](#LearningCharacterAgnosticMotionForMotionRetargetingIn2D)].

#### Human-Object Interaction

Gravity-Aware Monocular 3D Human-Object Reconstruction (ICCV 2021)
[[project](https://4dqv.mpi-inf.mpg.de/GraviCap)]
[[code](https://github.com/rishabhdabral/gravicap)]
[[data](https://drive.google.com/file/d/1qkcoWot9V4ydFTvilaLo5ahAW-fslnQQ/view?usp=drivesdk)]
[[paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Dabral_Gravity-Aware_Monocular_3D_Human-Object_Reconstruction_ICCV_2021_paper.pdf)]
[[arxiv](https://arxiv.org/abs/2108.08844)]
[[supplement](https://openaccess.thecvf.com/content/ICCV2021/supplemental/Dabral_Gravity-Aware_Monocular_3D_ICCV_2021_supplemental.pdf)]
[[video](https://www.youtube.com/watch?v=UsCOBNSBkqc)].

Neural Free-Viewpoint Performance Rendering under Complex Human-object Interactions (ACM MM 2021)
[[project](https://sunshinnnn.github.io/HOI-FVV)]
[[arxiv](https://arxiv.org/abs/2108.00362)]
[[video](https://www.youtube.com/watch?v=MPPABQORY2I)].

D3D-HOI: Dynamic 3D Human-Object Interactions from Videos (2021)
[[code](https://github.com/facebookresearch/d3d-hoi)]
[[arxiv](https://arxiv.org/abs/2108.08420)].

PHOSA: Perceiving 3D Human-Object Spatial Arrangements from a Single Image in the Wild (ECCV 2020)
[[project](https://jasonyzhang.com/phosa)]
[[code](https://github.com/facebookresearch/phosa)]
[[colab](https://colab.research.google.com/drive/1QIoL2g0jdt5E-vYKCIojkIz21j3jyEvo?usp=sharing)]
[[arxiv](https://arxiv.org/abs/2007.15649)]
[[video](https://www.youtube.com/watch?v=a5elX3x3Ssc)].

Angjoo Kanazawa: Perceiving 3D Human Interactions in the Wild
[[video](https://www.youtube.com/watch?v=CuSUtlCb5fE)].

#### Only 2D

<a name="LearningCharacterAgnosticMotionForMotionRetargetingIn2D"></a>
Learning Character-Agnostic Motion for Motion Retargeting in 2D (SIGGRAPH 2019)
[[project](https://motionretargeting2d.github.io)]
[[code](https://github.com/ChrisWu1997/2D-Motion-Retargeting)]
[[arxiv](https://arxiv.org/abs/1905.01680)]
[[video](https://www.youtube.com/watch?v=fR4h4OjZSdU)].

Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields (CVPR 2017)
[[code](https://github.com/ZheC/Realtime_Multi-Person_Pose_Estimation)]
[[arxiv](https://arxiv.org/abs/1611.08050)]
[[video](https://www.youtube.com/watch?v=OgQLDEAjAZ8&list=PLvsYSxrlO0Cl4J_fgMhj2ElVmGR5UWKpB)]
[[video](https://www.youtube.com/watch?v=pW6nZXeWlGM)].

AlphaPose
[[code](https://github.com/MVIG-SJTU/AlphaPose)].

### Statistical Body Models

Model parameter transfer
[[code](https://github.com/vchoutas/smplx/tree/master/transfer_model)] - 
code for converting model parameters of one model to another.

STAR: A Sparse Trained Articulated Human Body Regressor (ECCV2020)
[[project](https://star.is.tue.mpg.de)]
[[project](https://ps.is.mpg.de/publications/star-eccv-2020)]
[[code](https://github.com/ahmedosman/STAR)]
[[paper](https://ps.is.mpg.de/uploads_file/attachment/attachment/618/star_paper.pdf)]
[[supplement](https://ps.is.mpg.de/uploads_file/attachment/attachment/619/star_supmat.pdf)]
[[video](https://www.youtube.com/watch?v=JchovWRhrBs)].

Expressive Body Capture: 3D Hands, Face, and Body from a Single Image (CVPR 2019)
[[project](https://smpl-x.is.tue.mpg.de)]
[[code](https://github.com/vchoutas/smplify-x)]
[[code](https://github.com/vchoutas/smplx)]
[[paper](https://ps.is.mpg.de/uploads_file/attachment/attachment/497/SMPL-X.pdf)]
[[supplement](https://ps.is.mpg.de/uploads_file/attachment/attachment/498/SMPL-X-supp.pdf)]
[[poster](https://ps.is.mpg.de/uploads_file/attachment/attachment/517/smplx_poster.pdf)]
[[video](https://www.youtube.com/watch?v=XyXIEmapWkw)].

Embodied Hands: Modeling and Capturing Hands and Bodies Together (SIGGRAPH ASIA 2017)
[[project](https://mano.is.tue.mpg.de)]
[[project](https://ps.is.mpg.de/publications/embodiedhands)]
[[paper](https://ps.is.mpg.de/uploads_file/attachment/attachment/392/Embodied_Hands_SiggraphAsia2017.pdf)]

SMPL: A Skinned Multi-Person Linear Model (SIGGRAPH Asia 2015)
[[project](https://smpl.is.tue.mpg.de)]
[[paper](https://files.is.tue.mpg.de/black/papers/SMPL2015.pdf)]
[[video](https://www.youtube.com/watch?v=kuBlUyHeV5U)].

Exemplar-Based Statistical Model for Semantic Parametric Design of Human Body (2010)
[[paper](https://mewangcl.github.io/pubs/CiIHuman.pdf)].

A Statistical Model of Human Pose and Body Shape (Eurographics 2009)
[[project](https://gvvperfcapeva.mpi-inf.mpg.de/public/ScanDB)]

### Motion

MMFlow
[[code](https://github.com/open-mmlab/mmflow)] - 
open source optical flow toolbox based on PyTorch.
Not only, but mostly for human, so for now it is here.
*TODO*: add pertinent models from MMFlow here explicitly.

Physics-based Human Motion Estimation and Synthesis from Videos (ICCV 2021)
[[project](https://nv-tlabs.github.io/physics-pose-estimation-project-page)]
[[arxiv](https://arxiv.org/abs/2109.09913)].

Neural Monocular 3D Human Motion Capture with Physical Awareness (SIGGRAPH 2021)
[[project](https://vcai.mpi-inf.mpg.de/projects/PhysAware)]
[[paper](https://vcai.mpi-inf.mpg.de/projects/PhysAware/data/PhysAware.pdf)]
[[arxiv](https://arxiv.org/abs/2105.01057)]
[[video](https://www.youtube.com/watch?v=8JhUjzFAMJI)].

Motion Representations for Articulated Animation (CVPR 2021)
[[project](https://snap-research.github.io/articulated-animation)]
[[code](https://github.com/snap-research/articulated-animation)]
[[arxiv](https://arxiv.org/abs/2104.11280)]
[[video](https://www.youtube.com/watch?v=gpBYN8t8_yY)].

Contact and Human Dynamics from Monocular Video (ECCV 2020)
[[project](https://geometry.stanford.edu/projects/human-dynamics-eccv-2020)]
[[code](https://github.com/davrempe/contact-human-dynamics)]
[[paper](https://geometry.stanford.edu/projects/human-dynamics-eccv-2020/content/contact-and-dynamics-2020.pdf)]
[[arxiv](https://arxiv.org/abs/2007.11678)]
[[supplement](https://geometry.stanford.edu/projects/human-dynamics-eccv-2020/content/contact-and-dynamics-2020-supp.pdf)]
[[video](https://www.youtube.com/watch?v=qR9KW6JzXX4)].

MotioNet: 3D Human Motion Reconstruction from Monocular Video with Skeleton Consistency (ToG 2020)
[[project](https://rubbly.cn/publications/motioNet)]
[[code](https://github.com/Shimingyi/MotioNet)]
[[arxiv](https://arxiv.org/abs/2006.12075)]
[[video](https://www.youtube.com/watch?v=8YubchlzvFA)].

SFV: Reinforcement Learning of Physical Skills from Videos (SIGGRAPH Asia 2018)
[[project](https://xbpeng.github.io/projects/SFV/index.html)]
[[code](https://github.com/akanazawa/motion_reconstruction)]
[[paper](https://xbpeng.github.io/projects/SFV/2018_TOG_SFV.pdf)]
[[arxiv](https://arxiv.org/abs/1810.03599)]
[[video](https://www.youtube.com/watch?v=4Qg5I5vhX7Q)]
[[video](https://www.youtube.com/watch?v=_iXt7by4nU4)].

QuaterNet: A Quaternion-based Recurrent Model for Human Motion (BMVC 2018)
[[code](https://github.com/facebookresearch/QuaterNet)]
[[arxiv](https://arxiv.org/abs/1805.06485)]

On human motion prediction using recurrent neural networks (CVPR 2017)
[[code](https://github.com/una-dinosauria/human-motion-prediction)]
[[arxiv](https://arxiv.org/abs/1705.02445)]

A Deep Learning Framework for Character Motion Synthesis and Editing (SIGGRAPH 2016)
[[project](https://theorangeduck.com/page/deep-learning-framework-character-motion-synthesis-and-editing)]
[[paper](https://theorangeduck.com/media/uploads/motionsynthesis.pdf)]
[[video](https://www.youtube.com/watch?v=urf-AAIwNYk)]
[[video (VML Lab Seminar)](https://www.youtube.com/watch?v=R4C-7rcKmcQ)].

#### Motion Manifold

Constructing Human Motion Manifold with Sequential Networks (Computer Graphics Forum 2020)
[[project](https://motionlab.kaist.ac.kr/?page_id=5962)]
[[code](https://github.com/DK-Jang/human_motion_manifold)]
[[data](https://drive.google.com/file/d/1HNcgnCMOZ9p6WR-lsKhLOQhHbgOjZHhg/view?usp=sharing)]
[[arxiv](https://arxiv.org/abs/2005.14370)]
[[video](https://www.youtube.com/watch?v=DPXnidbmtvs)].

Learning Motion Manifolds with Convolutional Autoencoders (SIGGRAPH Asia 2015)
[[project](https://theorangeduck.com/page/learning-motion-manifolds-convolutional-autoencoders)]
[[paper](https://theorangeduck.com/media/uploads/motioncnn.pdf)]
[[supplement](https://theorangeduck.com/media/uploads/other_stuff/motioncnn.odp)]
[[video](https://www.youtube.com/watch?v=dLopOB6D9co)] - 
may be used for cleaning mocap data and in ```Person Pose and Shape Detection```[[local](#person-pose-and-shape)].

### Human Synthesys

#### Head

HeadGAN: One-shot Neural Head Synthesis and Editing (ICCV 2021)
[[project](https://michaildoukas.github.io/HeadGAN)]
[[arxiv](https://arxiv.org/abs/2012.08261)]
[[poster](https://www.dropbox.com/s/cs4b6wy5numb0yt/HeadGAN-poster.pdf?dl=1)]
[[video](https://www.youtube.com/watch?v=5eg85fi7Y5g)]
[[video](https://www.youtube.com/watch?v=Xo9IW3cMGTg)].

CoMA: Generating 3D faces using Convolutional Mesh Autoencoders ((ECCV) 2018)
[[project](https://coma.is.tue.mpg.de)]
[[project](https://ps.is.mpg.de/publications/coma)]
[[code](https://github.com/anuragranj/coma)]
[[code (PyTorch)](https://github.com/pixelite1201/pytorch_coma)]
[[code](https://ps.is.mpg.de/uploads_file/attachment/attachment/439/1285.pdf)]
[[code](https://arxiv.org/abs/1807.10267)]
[[code](https://ps.is.mpg.de/uploads_file/attachment/attachment/440/1285-supp.pdf)].

## View Synthesis

<img src="https://upload.wikimedia.org/wikipedia/commons/d/df/Container_01_KMJ.jpg" alt="container" title="Contains references to multiple projects" width="30" height="30"/>
New-View-Synthesis
[[project](https://github.com/visonpon/New-View-Synthesis)].
*TODO*: add papers from here.

Deep Image Spatial Transformation for Person Image Generation (CVPR 2020)
[[project]https://renyurui.github.io/GFLA-web)]
[[code](https://github.com/RenYurui/Global-Flow-Local-Attention)]
[[arxiv](https://arxiv.org/abs/2003.00696)]
[[video](https://www.youtube.com/watch?v=Ju0hBzCwsyU)].

Novel View Synthesis of Dynamic Scenes with Globally Coherent Depths from a Monocular Camera (CVPR 2020)
[[project](https://www-users.cse.umn.edu/~jsyoon/dynamic_synth)]
[[project](https://research.nvidia.com/publication/2020-06_Dynamic-Scene-View)]
[[arxiv](https://arxiv.org/abs/2004.01294)]
[[video](https://www.youtube.com/watch?v=S8_0V3fZIes)]
[[video](https://www.youtube.com/watch?v=kRKeoPkpPXM)].

Soft 3D Reconstruction for View Synthesis (SIGGRAPH Asia 2017)
[[project](https://ericpenner.github.io/soft3d)]
[[paper](https://ericpenner.github.io/soft3d/Soft_3D_Reconstruction.pdf)]
[[video](https://www.youtube.com/watch?v=szJBJ8oWrXI)].

Transformation-Grounded Image Generation Network for Novel 3D View Synthesis (CVPR 2017)
[[paroject](https://www.cs.unc.edu/~eunbyung/tvsn)]
[[code](https://github.com/silverbottlep/tvsn)]
[[paper](https://openaccess.thecvf.com/content_cvpr_2017/papers/Park_Transformation-Grounded_Image_Generation_CVPR_2017_paper.pdf)]
[[arxiv](https://arxiv.org/abs/1703.02921)].

Multi-Viewpoint-Image-generation
[[code](https://github.com/Chinmay26/Multi-Viewpoint-Image-generation)].

View Synthesis by Appearance Flow (ECCV 2016)
[[code](https://github.com/tinghuiz/appearance-flow)]
[[arxiv](https://arxiv.org/abs/1605.03557)].

[CVPR 2020] Novel View Synthesis Tutorial
[[project](https://nvlabs.github.io/nvs-tutorial-cvpr2020)]
[[video](https://www.youtube.com/watch?v=OEUHalxanuc)].

See ```Radiance Fields```[[local](#radiance-fields)].

## Radiance Fields

NeRF and related projects.

Instant Neural Graphics Primitives with a Multiresolution Hash Encoding
[[project](https://nvlabs.github.io/instant-ngp)]
[[code](https://github.com/NVlabs/instant-ngp)]
[[paper](https://nvlabs.github.io/instant-ngp/assets/mueller2022instant.pdf)]
[[video](https://nvlabs.github.io/instant-ngp/assets/mueller2022instant.mp4)].

BARF: Bundle-Adjusting Neural Radiance Fields (ICCV 2021)
[[project](https://chenhsuanlin.bitbucket.io/bundle-adjusting-NeRF)]
[[code](https://github.com/chenhsuanlin/bundle-adjusting-NeRF)]
[[paper](https://chenhsuanlin.bitbucket.io/bundle-adjusting-NeRF/paper.pdf)]
[[arxiv](https://arxiv.org/abs/2104.06405)]
[[supplement](https://chenhsuanlin.bitbucket.io/bundle-adjusting-NeRF/supplementary.pdf)]
[[video](https://www.youtube.com/watch?v=dCmCZs2Hpi0)].

GNeRF: GAN-based Neural Radiance Field without Posed Camera (ICCV 2021)
[[code](https://github.com/MQ66/gnerf)]
[[paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Meng_GNeRF_GAN-Based_Neural_Radiance_Field_Without_Posed_Camera_ICCV_2021_paper.pdf)]
[[arxiv](https://arxiv.org/abs/2103.15606)]
[[video](https://www.youtube.com/watch?v=r_Zf84kjNTM)].

NeRD: Neural Reflectance Decomposition from Image Collections (ICCV 2021)
[[project](https://markboss.me/publication/2021-nerd)]
[[code](https://github.com/cgtuebingen/NeRD-Neural-Reflectance-Decomposition)]
[[data](https://github.com/cgtuebingen/NeRD-Neural-Reflectance-Decomposition/blob/master/download_datasets.py)]
[[arxiv](https://arxiv.org/abs/2012.03918)]
[[video](https://www.youtube.com/watch?v=JL-qMTXw9VU)]
[[video](https://www.youtube.com/watch?v=IM9OgMwHNTI)].

Putting NeRF on a Diet: Semantically Consistent Few-Shot View Synthesis (ICCV 2021)
[[project](https://ajayj.com/dietnerf)]
[[code](https://github.com/ajayjain/DietNeRF)]
[[arxiv](https://arxiv.org/abs/2104.00677)]
[[video](https://www.youtube.com/watch?v=RF_3hsNizqw)]
[[video](https://www.youtube.com/watch?v=Isq2b3HnOoA)] - 
based on ```pixelNeRF: Neural Radiance Fields from One or Few Images```[[local](#pixelNeRF)].

Mip-NeRF: A Multiscale Representation for Anti-Aliasing Neural Radiance Fields (ICCV 2021)
[[project](https://jonbarron.info/mipnerf)]
[[code](https://github.com/google/mipnerf)]
[[arxiv](https://arxiv.org/abs/2103.13415)]
[[video](https://www.youtube.com/watch?v=EpH175PY1A0)].

Real-time Neural Radiance Caching for Path Tracing
[[project](https://research.nvidia.com/publication/2021-06_Real-time-Neural-Radiance)]
[[code](https://github.com/nvlabs/tiny-cuda-nn)]
[[demo](https://tom94.net/data/publications/mueller21realtime/interactive-viewer)]
[[paper](https://d1qx31qr3h6wln.cloudfront.net/publications/paper_4.pdf)]
[[arxiv](https://arxiv.org/abs/2106.12372)].

Baking Neural Radiance Fields for Real-Time View Synthesis
[[project](https://phog.github.io/snerg)]
[[code](https://github.com/google-research/google-research/tree/master/snerg)]
[[arxiv](https://arxiv.org/abs/2103.14645)]
[[video](https://www.youtube.com/watch?v=5jKry8n5YO8)].

Plenoxels: Radiance Fields without Neural Networks
[[project](https://alexyu.net/plenoxels)]
[[code](https://github.com/sxyu/svox2)]
[[arxiv](https://arxiv.org/abs/2112.05131)]
[[video](https://www.youtube.com/watch?v=KCDd7UFO1d0)].

NeRF in the Dark: High Dynamic Range View Synthesis from Noisy Raw Images
[[project](https://bmild.github.io/rawnerf)]
[[arxiv](https://arxiv.org/abs/2111.13679)]
[[video](https://www.youtube.com/watch?v=JtBS4KBcKVc)].

Neural Scene Flow Fields for Space-Time View Synthesis of Dynamic Scenes (CVPR 2021)
[[project](https://www.cs.cornell.edu/~zl548/NSFF)]
[[code](https://github.com/zl548/neural-scene-flow-fields)]
[[arxiv](https://arxiv.org/abs/2011.13084)]
[[video](https://www.youtube.com/watch?v=qsMIH7gYRCc)]. 
Neural Scene Flow Fields using pytorch-lightning 
[[code](https://github.com/kwea123/nsff_pl)].

<a name="NerfInTheWild"></a>
NeRF in the Wild: Neural Radiance Fields for Unconstrained Photo Collections (CVPR 2021)
[[project](https://nerf-w.github.io)]
[[paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Martin-Brualla_NeRF_in_the_Wild_Neural_Radiance_Fields_for_Unconstrained_Photo_CVPR_2021_paper.pdf)]
[[arxiv](https://arxiv.org/abs/2008.02268)]
[[video](https://www.youtube.com/watch?v=mRAKVQj5LRA)]. 
CSC2547 NeRF in the Wild Neural Radiance Fields for Unconstrained Photo Collections 
[[video](https://www.youtube.com/watch?v=BjXMXX9Pc6U)].

Unofficial implementation of NeRF (Neural Radiance Fields) using pytorch
[[code](https://github.com/kwea123/nerf_pl)] - 
including ```NeRF in the Wild```[[local](#NerfInTheWild)].

<a name="pixelNeRF"></a>
pixelNeRF: Neural Radiance Fields from One or Few Images
[[project](https://alexyu.net/pixelnerf)]
[[code](https://github.com/sxyu/pixel-nerf)]
[[data](https://drive.google.com/drive/folders/1PsT3uKwqHHD2bEEHkIXB99AlIjtmrEiR)]
[[arxiv](https://arxiv.org/abs/2012.02190)]
[[video](https://www.youtube.com/watch?v=voebZx7f32g)].

PlenOctrees for Real-time Rendering of Neural Radiance Fields (ICCV 2021)
[[project](https://alexyu.net/plenoctrees)]
[[code](https://github.com/sxyu/plenoctree)]
[[code](https://github.com/sxyu/volrend)]
[[arxiv](https://arxiv.org/abs/2103.14024)]
[[video](https://www.youtube.com/watch?v=obrmH1T5mfI)].

Neural Sparse Voxel Fields (NeurIPS 2020)
[[project](https://lingjie0206.github.io/papers/NSVF)]
[[code](https://github.com/facebookresearch/NSVF)]
[[arxiv](https://arxiv.org/abs/2007.11571)]
[[supplement](https://www.dropbox.com/s/sqsnl07fpfhwge2/nips_talk_10m_V3.pptm?dl=0)]
[[video](https://www.youtube.com/watch?v=RFqPwH7QFEI)].

NeRF++: Analyzing and Improving Neural Radiance Fields (2020)
[[code](https://github.com/Kai-46/nerfplusplus)]
[[arxiv](https://arxiv.org/abs/2010.07492)]
[[supplement](https://slides.games-cn.org/pdf/Games2021187KaiZhang.pdf)]
[[video](https://crossminds.ai/video/nerf-analyzing-and-improving-neural-radiance-fields-or-free-view-synthesis-or-vladlen-koltun-606f8d0875292b321dd09061)].

<a name="NeRFOrig"></a>
NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis (ECCV 2020)
[[project](https://www.matthewtancik.com/nerf)]
[[code](https://github.com/bmild/nerf)]
[[code (unafficial)](https://github.com/facebookresearch/NSVF)]
[[arxiv](https://arxiv.org/abs/2003.08934)]
[[video](https://www.youtube.com/watch?v=JuH79E8rdKc)].
Explaination 
[[video](https://www.youtube.com/watch?v=CRlN-cYFxTk)]
[[video](https://www.youtube.com/watch?v=dPWLybp4LL0)].

NeRF-pytorch
[[code](https://github.com/yenchenlin/nerf-pytorch)] - 
faithful PyTorch implementation of ```NeRF```[[local](#NeRFOrig)] 
that reproduces the results while running 1.3 times faster.

JaxNeRF
[[code](https://github.com/google-research/google-research/tree/master/jaxnerf)] - 
a JAX implementation of ```NeRF```[[local](#NeRFOrig)].

Understanding and Extending Neural Radiance Fields
[[blog](https://www.haikutechcenter.com/2021/06/understanding-and-extending-neural.html)]
[[video](https://www.youtube.com/watch?v=HfJpQCBTqZs)]
[[video](https://www.youtube.com/watch?v=nRyOzHpcr4Q)].

See ```View Synthesis```[[local](#view-synthesis)].

## Siren

Implicit Neural Representations with Periodic Activation Functions (NeurIPS 2020)
[[project](https://www.vincentsitzmann.com/siren)]
[[code](https://github.com/vsitzmann/siren)]
[[colab](https://colab.research.google.com/github/vsitzmann/siren/blob/master/explore_siren.ipynb)]
[[paper](https://proceedings.neurips.cc/paper/2020/file/53c04118df112c13a8c34b38343b9c10-Paper.pdf)]
[[arxiv](https://arxiv.org/abs/2006.09661)]
[[video](https://www.youtube.com/watch?v=Q2fLWGBeaiI)].

SIREN in Pytorch
[[code](https://github.com/lucidrains/siren-pytorch)].

Unofficial PyTorch implementation
[[code](https://github.com/dalmia/siren)].

Implicit Neural Representations with Periodic Activation Functions
[[blog](https://web.stanford.edu/~jnmartel/publication/sitzmann-2020-implicit)].

SIREN: Implicit Neural Representations with Periodic Activation Functions (Paper Explained)
[[video](https://www.youtube.com/watch?v=Q5g3p9Zwjrk)].

CSC2547 SIREN: Implicit Neural Representations with Periodic Activation Functions
[[video](https://www.youtube.com/watch?v=sM2QtUqfoXY)].

## Image Animation

First Order Motion Model for Image Animation
[[project](https://aliaksandrsiarohin.github.io/first-order-model-website)]
[[code](https://github.com/AliaksandrSiarohin/first-order-model)]
[[demo](https://colab.research.google.com/github/AliaksandrSiarohin/first-order-model/blob/master/demo.ipynb#scrollTo=UCMFMJV7K-ag)]
[[paper](https://papers.nips.cc/paper/2019/file/31c0b36aef265d9221af80872ceb62f9-Paper.pdf)]
[[supplement](https://papers.nips.cc/paper/2019/hash/31c0b36aef265d9221af80872ceb62f9-Abstract.html)]
[[video](https://www.youtube.com/watch?v=u-0cQ-grXBQ)].

## Tools

### Environments

*Possible application*: bots training.

OpenAI Gym
[[project](https://gym.openai.com)]
[[code](https://github.com/openai/gym)].

See ```DeepMind Environments```[[local](#deepmind-environments)].

#### OpenAI Environments

Multiagent emergence environments
[[code](https://github.com/openai/multi-agent-emergence-environments)].

Worldgen: Randomized MuJoCo [[local](#mujoco-physics)] environments
[[code](https://github.com/openai/mujoco-worldgen)].

Competitive Multi-Agent Environments
[[code](https://github.com/openai/multiagent-competition)].

#### Unity ML-Agents

Unity ML-Agents Toolkit
[[code](https://github.com/Unity-Technologies/ml-agents)].

How to generate game character behaviors using AI and ML (Unite Copenhagen 2019)
[[supplement](https://www.slideshare.net/unity3d/how-to-generate-game-character-behaviors-using-ai-and-ml-unite-copenhagen)]
[[video](https://www.youtube.com/watch?v=2M3ytOo7LQQ)].

##### AnimalAI

AnimalAI 3
[[code](https://github.com/mdcrosby/animal-ai)].

AnimalAI 2
[[code](https://github.com/beyretb/AnimalAI-Olympics)] - 
repo for version 2.0 and earlier.

#### Unreal Engine

MindMaker AI Plugin for Unreal Engine 4 & 5
[[code](https://github.com/krumiaa/MindMaker)].

##### Contests Environments

Google Research Football
[[code](https://github.com/google-research/football)]
[[arxiv](https://arxiv.org/abs/1907.11180)]. 
Example of agent 
[[code](https://github.com/ChintanTrivedi/rl-bot-football)].

#### Physics Environments

##### MuJoCo Physics

MuJoCo Physics
[[code](https://github.com/deepmind/mujoco)].

### Libs

<a name="Imaginaire"></a>
Imaginaire
[[code](https://github.com/NVlabs/imaginaire)]
[[blog](https://analyticsindiamag.com/guide-to-nvidia-imaginaire-gan-library-in-python)] - 
pytorch library that contains optimized implementation of several image and video synthesis methods developed at NVIDIA.

### Render

DIRT: a fast Differentiable Renderer for TensorFlow
[[code](https://github.com/pmh47/dirt)].

### Deployment

MMDeploy
[[code](https://github.com/open-mmlab/mmdeploy)] - 
open-source deep learning model deployment toolset.

### Optimization

MMRazor
[[code](https://github.com/open-mmlab/mmrazor)] - 
model compression toolkit for model slimming and AutoML.
## Other

microRTS
[[code](https://github.com/santiontanon/microrts)]
[[code](https://github.com/vwxyzjn/gym-microrts)] - 
small implementation of an RTS game, designed to perform AI research.

PyTorch Image Models
[[code](https://github.com/rwightman/pytorch-image-models)].

Griddly
[[project](https://griddly.readthedocs.io)]
[[code](https://github.com/SoftwareImpacts/SIMPAC-2021-6)] - 
cross platform grid-based research environment that is designed to be able to reproduce grid-world style games.

### Vision transformer

Vision Transformer and MLP-Mixer Architectures
[[code](https://github.com/google-research/vision_transformer)]
[[arxiv](https://arxiv.org/abs/2010.11929)]
[[blog](https://ai.googleblog.com/2020/12/transformers-for-image-recognition-at.html)].

Vision Transformer - Pytorch
[[code](https://github.com/lucidrains/vit-pytorch)].

Vision Transformer (ViT) - An image is worth 16x16 words \| Paper Explained
[[video](https://www.youtube.com/watch?v=j6kuz_NqkG0)].

Vision Transformer - Keras Code Examples!!
[[video](https://www.youtube.com/watch?v=i2_zJ0ANrw0)].

Vision Transformer for Image Classification
[[video](https://www.youtube.com/watch?v=HZ4j_U3FC94)].

Transformers in computer vision: ViT architectures, tips, tricks and improvements
[[blog](https://theaisummer.com/vision-transformer)].

How the Vision Transformer (ViT) works in 10 minutes: an image is worth 16x16 words
[[blog](https://theaisummer.com/transformers-computer-vision)].