import pandas as pd


    # df_train = df_train[df_train.target != 2]
    # df_train = df_train[df_train.target != 3]
    # df_valid = df_valid[df_valid.target != 2]
    # df_valid = df_valid[df_valid.target != 3]

    # df_train = df_train[:config.NUMBER_OF_SAMPLES]
    # df_valid = df_valid[:config.NUMBER_OF_SAMPLES]

    # df_train['sentence_1'][0] = "Do you like science fiction books?"
    # df_train['sentence_1'][1] = "Did you dance at the last concert you went to?"
    # df_train['sentence_1'][2] = "Do you read books on an eReader?"
    # df_train['sentence_1'][3] = "Do you think you might want to go camping this weekend?"

    # df_train['sentence_2'][0] = "I find them sort of depressing"
    # df_train['sentence_2'][1] = "I find them sort of depressing"
    # df_train['sentence_2'][2] = "I find them sort of depressing"
    # df_train['sentence_2'][3] = "I find them sort of depressing"

    # df_train['target'][0] = 0
    # df_train['target'][1] = 1
    # df_train['target'][2] = 2
    # df_train['target'][3] = 3

    # df_valid['sentence_1'][0] = "Do you like science fiction books?"
    # df_valid['sentence_1'][1] = "Did you dance at the last concert you went to?"
    # df_valid['sentence_1'][2] = "Do you read books on an eReader?"
    # df_valid['sentence_1'][3] = "Do you think you might want to go camping this weekend?"

    # df_valid['sentence_2'][0] = "I find them sort of depressing"
    # df_valid['sentence_2'][1] = "I find them sort of depressing"
    # df_valid['sentence_2'][2] = "I find them sort of depressing"
    # df_valid['sentence_2'][3] = "I find them sort of df_valid"
    # df_valid['target'][0] = 0
    # df_valid['target'][1] = 1
    # df_valid['target'][2] = 2
    # df_valid['target'][3] = 3

    # df_train = df_train.reset_index(drop=True)
    # df_valid = df_valid.reset_index(drop=True)

    # #for protype
    # df_train = df_train[:config.NUMBER_OF_SAMPLES]
    # df_valid = df_valid[:config.NUMBER_OF_SAMPLES]



args = parse_args()
        
    df_train = pd.read_csv(args.train)
    df_valid = pd.read_csv(args.valid)
    
    print(len(df_train), len(df_valid))

    if hasattr(args, 'sample_percentage'): 
        df_train = df_train[:int(len(df_train) * args.sample_percentage / 100)]
        df_valid = df_valid[:int(len(df_valid) * args.sample_percentage / 100)]

    print(len(df_train), len(df_valid))


        # # print('################')
    # # print(len(train_dataset))

    # # iterloader = iter(train_dataset)

    # # for i in range(0, len(train_dataset)):
    # #     item = next(iterloader)
    # #     print("iteration" + str(i))
    # #     print(item)

    # # print('################')

    # if __name__ == "__main__":
#     df = pd.read_csv(config.BOOLQ_TRAIN).dropna().reset_index(drop = True)
#     dset = CIRCADataset(
#         sentence_1=df.sentence_1.values,
#         sentence_2=df.sentence_2.values,
#         target=df.target.values
#         )
#     print(dset[1])


# # Print model's state_dict
    # print("Model's state_dict:")
    # for param_tensor in model.state_dict():
    #     print(param_tensor, "\t", model.state_dict()[param_tensor].size())

    # # Print optimizer's state_dict
    # print("Optimizer's state_dict:")
    # for var_name in optimizer.state_dict():
    #     print(var_name, "\t", optimizer.state_dict()[var_name])