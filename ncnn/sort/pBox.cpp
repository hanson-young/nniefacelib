#include"pBox.h"

void freepBox(struct pBox *pbox){
	if(pbox->pdata==NULL)cout<<"pbox is NULL!"<<endl;
	else 
		free(pbox->pdata);
	pbox->pdata = NULL;
	delete pbox;
}

void freepRelu(struct pRelu *prelu){
	if(prelu->pdata==NULL)cout<<"prelu is NULL!"<<endl;
	else 
		free(prelu->pdata);
	prelu->pdata = NULL;
	delete prelu;
}

void freeWeight(struct Weight *weight){
	if(weight->pdata==NULL)cout<<"weight is NULL!"<<endl;
	else 
		free(weight->pdata);
	weight->pdata = NULL;
	delete weight;
}

void pBoxShow(const struct pBox *pbox){
	if (pbox->pdata == NULL){
		cout << "pbox is NULL, please check it !" << endl;
		return;
	}
	cout << "the data is :" << endl;
	mydataFmt *p = pbox->pdata;
	//pbox->channel
	for (int channel = 0; channel < pbox->channel; channel++){
		cout << "the " << channel <<"th channel data is :"<< endl;
		//pbox->height
		for (int i = 0; i < pbox->height; i++){
			for (int k = 0; k < pbox->width; k++){
				cout << *p++ << " ";
			}
			cout << endl;
		}
	}
	p = NULL;
}

void pBoxShowE(const struct pBox *pbox,int channel, int row){
	if (pbox->pdata == NULL){
		cout << "the pbox is NULL, please check it !" << endl;
		return;
	}
	cout << "the data is :" << endl;
	mydataFmt *p = pbox->pdata + channel*pbox->width*pbox->height;
	//pbox->channel
	cout << "the " << channel <<"th channel data is :"<< endl;
	//pbox->height

	for (int i = 0; i < pbox->height; i++){
		if(i<0){
			for (int k = 0; k < pbox->width; k++){
				cout << *p++ << " ";
			}
			cout << endl;
		}
		else if(i==row){
			p += i*pbox->width;
			for (int k = 0; k < pbox->width; k++){
				if(k%4==0)cout<<endl;
				cout << *p++ << " ";
			}
			cout << endl;
		}
	}
	p = NULL;
}

void pReluShow(const struct pRelu *prelu){
	if (prelu->pdata == NULL){
		cout << "the prelu is NULL, please check it !" << endl;
		return;
	}
	cout << "the data is :" << endl;
	mydataFmt *p = prelu->pdata;
	for (int i = 0; i < prelu->width; i++){
			cout << *p++ << " ";
		}
		cout << endl;
	p = NULL;
}

void weightShow(const struct Weight *weight){
	if (weight->pdata == NULL){
		cout << "the weight is NULL, please check it !" << endl;
		return;
	}
	cout << "the weight data is :" << endl;
	mydataFmt *p = weight->pdata;
	for (int channel = 0; channel < weight->selfChannel; channel++){
		cout << "the " << channel <<"th channel data is :"<< endl;
		for (int i = 0; i < weight->lastChannel; i++){
			for (int k = 0; k < weight->kernelSize*weight->kernelSize; k++){
				cout << *p++ << " ";
			}
			cout << endl;
		}
	}
	p = NULL;
}