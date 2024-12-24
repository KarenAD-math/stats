#ifndef SVMOVERLOADS_H
#define SVMOVERLOADS_H

#include "svm.cpp"

//
// Interface functions
//
svm_model *svm_train_qt(const svm_problem *prob, const svm_parameter *param)
{
    svm_model *model = Malloc(svm_model,1);
    model->param = *param;
    model->free_sv = 0;	// XXX

    if(param->svm_type == ONE_CLASS ||
        param->svm_type == EPSILON_SVR ||
        param->svm_type == NU_SVR)
    {
        // regression or one-class-svm
        model->nr_class = 2;
        model->label = NULL;
        model->nSV = NULL;
        model->probA = NULL; model->probB = NULL;
        model->prob_density_marks = NULL;
        model->sv_coef = Malloc(double *,1);

        decision_function f = svm_train_one(prob,param,0,0);
        model->rho = Malloc(double,1);
        model->rho[0] = f.rho;

        int nSV = 0;
        int i;
        for(i=0;i<prob->l;i++)
            if(fabs(f.alpha[i]) > 0) ++nSV;
        model->l = nSV;
        model->SV = Malloc(svm_node *,nSV);
        model->sv_coef[0] = Malloc(double,nSV);
        model->sv_indices = Malloc(int,nSV);
        int j = 0;
        for(i=0;i<prob->l;i++)
            if(fabs(f.alpha[i]) > 0)
            {
                model->SV[j] = prob->x[i];
                model->sv_coef[0][j] = f.alpha[i];
                model->sv_indices[j] = i+1;
                ++j;
            }

        if(param->probability &&
            (param->svm_type == EPSILON_SVR ||
             param->svm_type == NU_SVR))
        {
            model->probA = Malloc(double,1);
            model->probA[0] = svm_svr_probability(prob,param);
        }
        else if(param->probability && param->svm_type == ONE_CLASS)
        {
            int nr_marks = 10;
            double *prob_density_marks = Malloc(double,nr_marks);

            if(svm_one_class_probability(prob,model,prob_density_marks) == 0)
                model->prob_density_marks = prob_density_marks;
            else
                free(prob_density_marks);
        }

        free(f.alpha);
    }
    else
    {
        // classification
        int l = prob->l;
        int nr_class;
        int *label = NULL;
        int *start = NULL;
        int *count = NULL;
        int *perm = Malloc(int,l);

        // group training data of the same class
        svm_group_classes(prob,&nr_class,&label,&start,&count,perm);
        if(nr_class == 1)
            info("WARNING: training data in only one class. See README for details.\n");

        svm_node **x = Malloc(svm_node *,l);
        int i;
        for(i=0;i<l;i++)
            x[i] = prob->x[perm[i]];

        // calculate weighted C

        double *weighted_C = Malloc(double, nr_class);
        for(i=0;i<nr_class;i++)
            weighted_C[i] = param->C;
        for(i=0;i<param->nr_weight;i++)
        {
            int j;
            for(j=0;j<nr_class;j++)
                if(param->weight_label[i] == label[j])
                    break;
            if(j == nr_class)
                fprintf(stderr,"WARNING: class label %d specified in weight is not found\n", param->weight_label[i]);
            else
                weighted_C[j] *= param->weight[i];
        }

        // train k*(k-1)/2 models (One vs One?)

        bool *nonzero = Malloc(bool,l);
        for(i=0;i<l;i++)
            nonzero[i] = false;
        decision_function *f = Malloc(decision_function,nr_class*(nr_class-1)/2);

        double *probA=NULL,*probB=NULL;
        if (param->probability)
        {
            probA=Malloc(double,nr_class*(nr_class-1)/2);
            probB=Malloc(double,nr_class*(nr_class-1)/2);
        }

        //prepare for training
        int p = 0;
        for(i=0;i<nr_class;i++)
            for(int j=i+1;j<nr_class;j++)
            {
                //prepare the sub-problem for class i and j
                svm_problem sub_prob;


                int si = start[i], sj = start[j];
                int ci = count[i], cj = count[j];
                sub_prob.l = ci+cj;
                sub_prob.x = Malloc(svm_node *,sub_prob.l);
                sub_prob.y = Malloc(double,sub_prob.l);

                int k;

                for(k=0;k<ci;k++)
                {
                    sub_prob.x[k] = x[si+k];
                    sub_prob.y[k] = +1;
                }

                for(k=0;k<cj;k++)
                {
                    sub_prob.x[ci+k] = x[sj+k];
                    sub_prob.y[ci+k] = -1;
                }

                //only if probability
                if(param->probability)
                    svm_binary_svc_probability(&sub_prob,param,weighted_C[i],weighted_C[j],probA[p],probB[p]);

                //train the SVM for the given sub-problem, returns a decision function
                f[p] = svm_train_one(&sub_prob,param,weighted_C[i],weighted_C[j]);


                for(k=0;k<ci;k++)
                    if(!nonzero[si+k] && fabs(f[p].alpha[k]) > 0)
                        nonzero[si+k] = true;
                for(k=0;k<cj;k++)
                    if(!nonzero[sj+k] && fabs(f[p].alpha[ci+k]) > 0)
                        nonzero[sj+k] = true;
                free(sub_prob.x);
                free(sub_prob.y);
                ++p;
            }

        // build output

        model->nr_class = nr_class;

        model->label = Malloc(int,nr_class);
        for(i=0;i<nr_class;i++)
            model->label[i] = label[i];

        model->rho = Malloc(double,nr_class*(nr_class-1)/2);
        for(i=0;i<nr_class*(nr_class-1)/2;i++)
            model->rho[i] = f[i].rho;

        if(param->probability)
        {
            model->probA = Malloc(double,nr_class*(nr_class-1)/2);
            model->probB = Malloc(double,nr_class*(nr_class-1)/2);
            for(i=0;i<nr_class*(nr_class-1)/2;i++)
            {
                model->probA[i] = probA[i];
                model->probB[i] = probB[i];
            }
        }
        else
        {
            model->probA=NULL;
            model->probB=NULL;
        }
        model->prob_density_marks=NULL;	// for one-class SVM probabilistic outputs only

        int total_sv = 0;
        int *nz_count = Malloc(int,nr_class);
        model->nSV = Malloc(int,nr_class);
        for(i=0;i<nr_class;i++)
        {
            int nSV = 0;
            for(int j=0;j<count[i];j++)
                if(nonzero[start[i]+j])
                {
                    ++nSV;
                    ++total_sv;
                }
            model->nSV[i] = nSV;
            nz_count[i] = nSV;
        }

        info("Total nSV = %d\n",total_sv);

        model->l = total_sv;
        model->SV = Malloc(svm_node *,total_sv);
        model->sv_indices = Malloc(int,total_sv);
        p = 0;
        for(i=0;i<l;i++)
            if(nonzero[i])
            {
                model->SV[p] = x[i];
                model->sv_indices[p++] = perm[i] + 1;
            }

        int *nz_start = Malloc(int,nr_class);
        nz_start[0] = 0;
        for(i=1;i<nr_class;i++)
            nz_start[i] = nz_start[i-1]+nz_count[i-1];

        model->sv_coef = Malloc(double *,nr_class-1);
        for(i=0;i<nr_class-1;i++)
            model->sv_coef[i] = Malloc(double,total_sv);

        p = 0;
        for(i=0;i<nr_class;i++)
            for(int j=i+1;j<nr_class;j++)
            {
                // classifier (i,j): coefficients with
                // i are in sv_coef[j-1][nz_start[i]...],
                // j are in sv_coef[i][nz_start[j]...]

                int si = start[i];
                int sj = start[j];
                int ci = count[i];
                int cj = count[j];

                int q = nz_start[i];
                int k;
                for(k=0;k<ci;k++)
                    if(nonzero[si+k])
                        model->sv_coef[j-1][q++] = f[p].alpha[k];
                q = nz_start[j];
                for(k=0;k<cj;k++)
                    if(nonzero[sj+k])
                        model->sv_coef[i][q++] = f[p].alpha[ci+k];
                ++p;
            }

        free(label);
        free(probA);
        free(probB);
        free(count);
        free(perm);
        free(start);
        free(x);
        free(weighted_C);
        free(nonzero);
        for(i=0;i<nr_class*(nr_class-1)/2;i++)
            free(f[i].alpha);
        free(f);
        free(nz_count);
        free(nz_start);
    }
    return model;
}

#endif // SVMOVERLOADS_H
