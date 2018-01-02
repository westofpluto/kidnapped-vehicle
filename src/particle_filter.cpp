/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
    //
    // Begine by creating the normal distributions we need
    // 
    default_random_engine gen;
    normal_distribution<double> dist_x(x, std[0]);
    normal_distribution<double> dist_y(y, std[1]);
    normal_distribution<double> dist_theta(theta, std[2]);

    //
    // Set number of particle
    //
    num_particles=500;

    //
    // create the particles
    //

    cout << "creating this many particles: " << num_particles << "\n";

    for (int i=0;i<num_particles;i++) {
        Particle p=Particle();
        p.id=i;
        p.x=dist_x(gen);
        p.y=dist_y(gen);
        p.theta=dist_theta(gen);
        p.weight=1.0;
        particles.push_back(p);
    }
    is_initialized=true;
    cout << "Finished creating this many particles: " << particles.size() << "\n";
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
    default_random_engine gen;
    double xn,yn,thetan;
    double xf,yf,thetaf;
    int i;
 
    for (i=0;i<num_particles;i++) {
        normal_distribution<double> dist_x(particles[i].x, std_pos[0]);
        normal_distribution<double> dist_y(particles[i].y, std_pos[1]);
        normal_distribution<double> dist_theta(particles[i].theta, std_pos[2]);
        if (fabs(yaw_rate) < 0.000001) {
            xn=dist_x(gen);
            yn=dist_y(gen);
            thetan=dist_theta(gen);
            thetaf=thetan;
            xf=xn + velocity*delta_t*cos(thetan);
            yf=yn + velocity*delta_t*sin(thetan);
            particles[i].x=xf;
            particles[i].y=yf;
            particles[i].theta=thetaf;
        } else { 
            xn=dist_x(gen);
            yn=dist_y(gen);
            thetan=dist_theta(gen);
            thetaf=thetan + yaw_rate*delta_t;
            xf=xn + velocity*(sin(thetaf) - sin(thetan))/yaw_rate;
            yf=yn + velocity*(cos(thetan) - cos(thetaf))/yaw_rate;
            particles[i].x=xf;
            particles[i].y=yf;
            particles[i].theta=thetaf;
        }
    }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
    int i,j,min_id;
    int num_pred=predicted.size();
    int num_obs=observations.size();
    double d,min_dist;

    for (i=0;i<num_obs;i++) {
        min_dist=1.0E+20;  // any super-large number will do
        min_id=-1; 
        LandmarkObs obs=observations[i];
         
        for (j=0;j<num_pred;j++) {
            LandmarkObs pred=predicted[j];
            d=dist(pred.x,pred.y,obs.x,obs.y);
            if (d<min_dist) {
                min_dist=d;
                min_id=pred.id;
            }
        }
        observations[i].id=min_id;
    }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
    int i,j,k;
    LandmarkObs obs,obsmap,predmap;
    double cth,sth,d;
    double sig_x,sig_y;

    sig_x=std_landmark[0];
    sig_y=std_landmark[1];

    for (i=0;i<num_particles;i++) {
        Particle p=particles[i];
        cth=cos(p.theta);
        sth=sin(p.theta);
        //
        // predict measurements to all map landmarks within range
        // This results in a vector "predicted_map" where the measurements are in map coordinate system
        //
        std::vector<LandmarkObs> predicted_map;
        for (j=0;j<map_landmarks.landmark_list.size();j++) {
            d=dist(p.x,p.y,(double)map_landmarks.landmark_list[j].x_f,(double)map_landmarks.landmark_list[j].y_f);
            if (d <= sensor_range) {
                predicted_map.push_back(LandmarkObs{ map_landmarks.landmark_list[j].id_i, (double)map_landmarks.landmark_list[j].x_f, (double)map_landmarks.landmark_list[j].y_f }); 
            }
        }

        //
        // Now convert observations (in vehicle coordinate system) to new vector observations_map in map coordainet system
        //
        std::vector<LandmarkObs> observations_map;
        for (j=0;j<observations.size();j++) {
            obs=observations[j];
            //
            // transform to map coords
            //
            obsmap.id=obs.id; 
            obsmap.x=cth*obs.x - sth*obs.y + p.x;
            obsmap.y=sth*obs.x + cth*obs.y + p.y;
            observations_map.push_back(obsmap);
        }

        //
        // Now do data association so we know which measurement belongs to which landmark
        //
        dataAssociation(predicted_map,observations_map);

        //
        // now compute weights
        //
        //cout << "Getting weight for particle " << i << "\n";
        particles[i].weight=1.0;
        for (j=0;j<observations_map.size();j++) {
            obsmap=observations_map[j];
            int theid=obsmap.id;
            // find predicted measurement
            for (k=0;k<predicted_map.size();k++) {
                if (predicted_map[k].id == theid) {
                    predmap=predicted_map[k];
                    break;
                }
            }
            //cout <<     "Obs[" << j << "] has id " << predmap.id << "\n";
            double tmp=(predmap.x-obsmap.x)*(predmap.x-obsmap.x)/(2.0*sig_x*sig_x);
            tmp +=(predmap.y-obsmap.y)*(predmap.y-obsmap.y)/(2.0*sig_y*sig_y);
            double prob=exp(-tmp)/(2.0*M_PI*sig_x*sig_y);
            //cout <<     "prob is " << prob << "\n";
            particles[i].weight *= prob;
        }
        //cout <<     "particles[i].weight = " << particles[i].weight << "\n";
    }
}

void ParticleFilter::resample() {
    int i;
    double max_weight=0.0;
    default_random_engine gen;

    for (i=0;i<num_particles;i++) {
        if (particles[i].weight > max_weight) {
            max_weight=particles[i].weight;
        }
    }

    uniform_int_distribution<int> index_dist(0, num_particles-1);
    int index=index_dist(gen);
    uniform_real_distribution<double> beta_dist(0.0, 2.0*max_weight);
    double beta=0.0;
    
    std::vector<Particle> new_particles;
    for (i=0;i<num_particles;i++) {
        beta += beta_dist(gen);
        while (beta > particles[index].weight) {
            beta -= particles[index].weight;
            index++;
            if (index>=num_particles) {
                index=0;
            }
        }
        new_particles.push_back(particles[index]);
    }
    particles=new_particles;
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;

    return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
