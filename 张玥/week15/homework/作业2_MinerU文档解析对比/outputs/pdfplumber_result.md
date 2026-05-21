# pdfplumber 解析结果摘要

- PDF 文件：`D:\AI_study_env\files\study\Week15\模型论文\2509-MinerU2.5.pdf`
- 总页数：57
- 本次抽取页数：3
- 抽取文本长度：4658

## 第 1 页

MinerU2.5: A Decoupled Vision-Language Model for Efficient
High-Resolution Document Parsing
JunboNiu1,2∗,ZhengLiu1,2∗,ZhuangchengGu1∗,BinWang1∗‡,LinkeOuyang1∗
ZhiyuanZhao1∗,TaoChu1∗,TianyaoHe1∗,FanWu1∗,QintongZhang1,2∗,ZhenjiangJin1∗
GuangLiang1,RuiZhang1,WenzhengZhang1,2,YuanQu1,ZhifeiRen1,YuefengSun1
YuanhongZheng1,DongshengMa1,ZiruiTang1,3,BoyuNiu1,3,ZiyangMiao1,HejunDong1
SiyiQian1,2,JunyuanZhang1,JingzhouChen1,2,FangdongWang1,XiaomengZhao1,LiqunWei1
WeiLi1,ShashaWang1, RuiliangXu1, YuanyuanCao1, LuChen1, QianqianWu1, HuaiyuGu1
LindongLu1,KemingWang1, DechenLin1, GuanlinShen1, XuanheZhou1,3,LinfengZhang3
YuhangZang1,XiaoyiDong1,JiaqiWang1,BoZhang1,LeiBai1,PeiChu1,WeijiaLi1,JiangWu1
LijunWu1,ZhenxiangLi1,GuangyuWang1,ZhongyingTu1,ChaoXu1,KaiChen1
YuQiao1,BowenZhou1,DahuaLin1(cid:0),WentaoZhang1,2(cid:0),ConghuiHe1(cid:0)
1ShanghaiArtificialIntelligenceLaboratory,2PekingUniversity,3ShanghaiJiaoTongUniversity
We introduce MinerU2.5, a 1.2B-parameter document parsing vision-language model that
achievesstate-of-the-artrecognitionaccuracywhilemaintainingexceptionalcomputational
efficiency. Ourapproachemploysacoarse-to-fine,two-stageparsingstrategythatdecouples
globallayoutanalysisfromlocalcontentrecognition. Inthefirststage,themodelperforms
efficientlayoutanalysisondownsampledimagestoidentifystructuralelements,circumvent-
ing the computational overhead of processing high-resolution inputs. In the second stage,
guided by the global layout, it performs targeted content recognition on native-resolution
cropsextractedfromtheoriginalimage,preservingfine-graineddetailsindensetext,complex
formulas,andtables. Tosupportthisstrategy,wedevelopedacomprehensivedataengine
thatgeneratesdiverse,large-scaletrainingcorporaforbothpretrainingandfine-tuning. Ulti-
mately,MinerU2.5demonstratesstrongdocumentparsingability,achievingstate-of-the-art
performanceonmultiplebenchmarks,surpassingbothgeneral-purposeanddomain-specific
modelsacrossvariousrecognitiontasks,whilemaintainingsignificantly

## 第 2 页

MinerU2.5: ADecoupledVision-LanguageModelforEfficientHigh-ResolutionDocumentParsing
Contents
1 Introduction 4
2 RelatedWork 5
2.1 TraditionalPipelines . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 5
2.2 General-PurposeVisionLanguageModels . . . . . . . . . . . . . . . . . . . . . . . . . . 6
2.3 Domain-SpecificVisionLanguageModels . . . . . . . . . . . . . . . . . . . . . . . . . . . 6
3 MinerU2.5 6
3.1 ModelArchitecture . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 6
3.2 Two-StageParsingStrategy . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 7
3.3 TrainingRecipe . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 8
3.3.1 Stage0-ModalityAlignment . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 8
3.3.2 Stage1-DocumentParsingPre-training . . . . . . . . . . . . . . . . . . . . . . . . 9
3.3.3 Stage2-DocumentParsingFine-tuning . . . . . . . . . . . . . . . . . . . . . . . . 9
3.3.4 DataAugmentationStrategies . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 10
3.4 ModelDeployment . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 10
4 DataEngine 11
4.1 OverallWorkflow . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 11
4.1.1 DataCuration . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 11
4.1.2 Pre-trainingDatasetPreparation . . . . . . . . . . . . . . . . . . . . . . . . . . . . 12
4.1.3 Fine-tuningDatasetConstruction . . . . . . . . . . . . . . . . . . . . . . . . . . . 13
4.2 TaskReformulationandEnhancement . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 13
4.2.1 LayoutAnalysis. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 13
4.2.2 FormulaRecognition . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 15
4.2.3 TableRecognition . . . . . . . . . . . . . . . . . . . . . . . . . . . . 

## 第 3 页

MinerU2.5: ADecoupledVision-LanguageModelforEfficientHigh-ResolutionDocumentParsing
A.3.3 Layout&OCR . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 53
B PromptDetails 56
B.1 LayoutDetection . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 56
B.2 TextRecognition . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 56
B.3 FormulaRecognition . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 56
B.4 TableRecognition . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 57
3
