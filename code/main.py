import numpy as np
initalprobabilities={}
transitionprobabilities={}
emissionprobabilities={}

initalcounts={}
transitioncounts={}
emissioncount={}

tagcounts={}#count of all tags in given training set
alltags=[]#this is just temporary variable
falsecounts=0
allcount=0
def add(key):#tag counter
    if key in tagcounts:
        tagcounts[key]+=1
    else:
        tagcounts[key]=1
def comparetwolist(x, y):#compare two list and return number of different elements
    count=0
    for i in range(0,len(x)):
        if x[i]!=y[i]:
            count+=1
    return count

############################Task 1: Build a Bigram Hidden Markov Model (HMM)############################
file=open("metu.txt", "r", encoding="utf8")
for i in range(0,3960):#read training set
    line=file.readline()
    line=line.split()
    firsttag=line[0].split("/")
    alltags.append(firsttag[1].lower())
    key=firsttag[1]+"|"+"<s>"
    key=key.lower()

    if key in initalprobabilities:#construct initial counts
        initalprobabilities[key]+=1
    else:
        initalprobabilities[key]=1
    for i in range(0,len(line)-1):#construct emission and transition counts
        current=line[i].split("/")
        next=line[i+1].split("/")
        keyfortransitionprobability=next[1]+"|"+current[1]
        alltags.append(next[1].lower())
        alltags.append(current[1].lower())

        add(current[1].lower())
        keyfortransitionprobability=keyfortransitionprobability.lower()
        keyforemissionprobability=current[0]+"|"+current[1]

        alltags.append(current[1].lower())
        keyforemissionprobability=keyforemissionprobability.lower()
        if keyfortransitionprobability in transitionprobabilities:
            transitionprobabilities[keyfortransitionprobability]+=1
        else:
            transitionprobabilities[keyfortransitionprobability]=1
        if keyforemissionprobability in emissionprobabilities:
            emissionprobabilities[keyforemissionprobability]+=1
        else:
            emissionprobabilities[keyforemissionprobability]=1
    lastkeyforemisionprobability=line[-1].split("/")#used for bounding problem of sentence
    add(lastkeyforemisionprobability[1].lower())
    lastkeyforemisionprobability=lastkeyforemisionprobability[0]+"|"+lastkeyforemisionprobability[1]
    lastkeyforemisionprobability=lastkeyforemisionprobability.lower()
    if lastkeyforemisionprobability in emissionprobabilities:
        emissionprobabilities[lastkeyforemisionprobability] += 1
    else:
        emissionprobabilities[lastkeyforemisionprobability] = 1
#two different dictionaries for counts and probabilities
initalcounts=dict(initalprobabilities)
transitioncounts=dict(transitionprobabilities)
emissioncount=dict(emissionprobabilities)
sizeofinitalprobabilities=sum(initalprobabilities.values())
sizeoftransitionprobabilities=sum(transitionprobabilities.values())
sizeofemissionprobabilities=sum(emissionprobabilities.values())
for word in initalprobabilities:#construct initial probablities
    initalprobabilities[word]/=sizeofinitalprobabilities
for word in transitionprobabilities:#construct transition probablities
    transitionprobabilities[word]/=sizeoftransitionprobabilities
for word in emissionprobabilities:#construct emission probablities
    emissionprobabilities[word]/=sizeofemissionprobabilities

########################################End of Task 1###############################################
##########################################Task 2####################################################
#Calculate the most likely tag sequence and  trace the back pointers to find the most likely tag
#sequence from the end of the sentence till the beginning#
def computeprobability(line,currentlinenumber):
    predictedtags=[]
    realtags=[]
    words=[]
    line = line.lower()
    line = line.split()




    firstword=line[0].split("/")
    firsttag=firstword[1]
    w, h = len(line)+1, len(set(alltags))+1;
    matrix = [[0 for x in range(w)] for y in range(h)]#probablity matrix
    matrix2=[["x" for x in range(w)] for y in range(h)]#matrix of last step tag for viterbi
    taglist=list(set(alltags))
    for i in range (len(set(alltags))):
        key=taglist[i]+"|"+"<s>"
        valueofinitial=0
        if key in initalcounts:#first tag probablity calculating
            valueofinitial=(1+initalcounts[key])/(sum(initalcounts.values())+len(initalcounts))
        else:
            valueofinitial=(1+0)/(sum(initalcounts.values())+len(initalcounts))#smoothing for unseen tag of inital word's tag
        key=firstword[0]+"|"+taglist[i]
        valueofemission=0
        if key in emissioncount:#emission probablity of first tag
            valueofemission=(1+emissioncount[key])/(tagcounts[taglist[i]]+len(emissioncount.keys()))
        else:
            valueofemission=(1+0)/(tagcounts[taglist[i]]+len(emissioncount.keys()))#smoothing
        matrix[i+1][0]=taglist[i]#first column contains tags of training set
        matrix[i+1][1]=np.log2(valueofinitial*valueofemission)#add calculated probablities to matrix of probablities
        matrix2[i+1][0]=taglist[i]
    for j in range(len(line)):#add words to first line
        word=line[j].split("/")
        matrix[0][j+1]=word[0]
        words.append(word[0])
        realtags.append(word[1])

    for j in range(2,len(line)+1):
        for i in range(1,len(set(alltags))+1):#traverse up-down for each cell
            firstag=matrix[i][0]
            word=matrix[0][j]#word
            values=[]
            for k in range(1,len(set(alltags))+1):
                #for each value of matrix' cell multiplies previous value,emission value and transition value
                emissionvalue=0
                transitionvalue=0
                prevvalue = matrix[k][j - 1]
                keyfortransitionprobability=matrix[k][0]+"|"+firstag
                keyforemissionprobability=word+"|"+matrix[i][0]

                #print(keyfortransitionprobability,keyforemissionprobability,prevvalue)
                if keyfortransitionprobability in transitioncounts:
                    transitionvalue=(1+transitioncounts[keyfortransitionprobability])/(tagcounts[firsttag]+len(transitioncounts.keys()))
                else:#smoothing
                    transitionvalue=(1+0)/(tagcounts[firsttag]+len(transitioncounts.keys()))
                if keyforemissionprobability in emissioncount:
                    emissionvalue=(1+emissioncount[keyforemissionprobability])/(tagcounts[matrix[i][0]]+len(emissioncount.keys()))
                else:#smoothing
                    emissionvalue=(1+0)/(tagcounts[matrix[i][0]]+len(emissioncount.keys()))
                values.append(np.log2(transitionvalue)+np.log2(emissionvalue)+prevvalue)

            matrix[i][j]=max(values)
            #find maximum value of tag and select it
            matrix2[i][j]=taglist[values.index(max(values))]
            #adding matrix2 means stores the previous steps
            values.clear()
    lasttag = matrix2[taglist.index("punc") + 1][w - 1]#this always select the last tag is punc
    predictedtags.append("punc")
    predictedtags.append(lasttag)
    for j in range(0,w-3):#viterbi algorithm
        currenttag=matrix2[taglist.index(lasttag)+1][w-j-2]
        #go back and find previous tags
        lasttag=currenttag
        predictedtags.append(currenttag)
    predictedtags.reverse()

    # you can easily print my matrices
    # print(np.matrix(matrix))
    # print(line)
    # print(np.matrix(matrix2))
    # print(predictedtags,realtags)

    #print("Real Tags for given sentence:",realtags)
    #print("Predicted Tags for given sentence:",predictedtags)
    wr=""
    #predictedtags[-2]="verb"
    #^generally second last tag is verb in Turkish.You can run last line and it will increase the accuracy by 4.
    words[0]=words[0].capitalize()
    for i in range(len(words)):
        wr=wr+words[i]+"/"+predictedtags[i]+" "
    print("Line:"+str(currentlinenumber)+":"+wr)
    return comparetwolist(predictedtags,realtags),len(line)
########################################End of Task 2###############################################
for i in range(0,1699):
    lenoffalsetags,lenofline=computeprobability(file.readline(),i+1)
    falsecounts+=lenoffalsetags
    allcount+=lenofline
print(initalprobabilities)
print(emissionprobabilities)
print(transitionprobabilities)
print(initalcounts)
print(emissioncount)
print(transitioncounts)
##########################################Task 3:Calculate Accuracy####################################################
print("accuracy:%" + str(100 * ((allcount - falsecounts) / allcount)))
########################################End of Task 3###############################################
file.close()

