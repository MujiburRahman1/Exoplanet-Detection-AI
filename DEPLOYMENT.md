# Exoplanet Detection AI - Deployment Guide

## 🚀 Quick Deployment Options

### 1. Heroku Deployment

#### Prerequisites:
- Heroku CLI installed
- Git repository pushed to GitHub

#### Steps:
```bash
# Install Heroku CLI
# Login to Heroku
heroku login

# Create Heroku app
heroku create your-exoplanet-app-name

# Set environment variables (optional)
heroku config:set FLASK_ENV=production

# Deploy
git push heroku main

# Open app
heroku open
```

### 2. Railway Deployment

#### Steps:
1. Go to [Railway.app](https://railway.app)
2. Connect your GitHub repository
3. Select the repository: `Exoplanet-Detection-AI`
4. Railway will auto-detect Python app
5. Deploy automatically

### 3. Render Deployment

#### Steps:
1. Go to [Render.com](https://render.com)
2. Connect GitHub account
3. Create new Web Service
4. Connect repository: `Exoplanet-Detection-AI`
5. Build Command: `pip install -r requirements.txt`
6. Start Command: `gunicorn wsgi:app`

### 4. Vercel Deployment

#### Steps:
1. Go to [Vercel.com](https://vercel.com)
2. Import GitHub repository
3. Configure as Python project
4. Deploy

## 🔧 Configuration Files

### Required Files (Already Created):
- ✅ `Procfile` - Heroku deployment
- ✅ `runtime.txt` - Python version
- ✅ `wsgi.py` - WSGI configuration
- ✅ `requirements.txt` - Dependencies

### Environment Variables (Optional):
```bash
FLASK_ENV=production
PORT=5000
```

## 📁 Project Structure
```
Exoplanet-Detection-AI/
├── src/
│   ├── web/
│   │   └── app.py          # Main Flask app
│   ├── models/
│   │   └── exoplanet_classifier.py
│   └── data/
│       └── data_loader.py
├── templates/              # HTML templates
├── static/                # CSS/JS files
├── data/                  # NASA datasets
├── models/                # Trained models
├── Procfile               # Heroku deployment
├── wsgi.py               # WSGI configuration
├── requirements.txt      # Python dependencies
└── runtime.txt           # Python version
```

## 🎯 Deployment Checklist

### Before Deployment:
- [ ] All files committed to Git
- [ ] Repository pushed to GitHub
- [ ] Requirements.txt updated
- [ ] Procfile created
- [ ] WSGI configuration ready

### After Deployment:
- [ ] App URL accessible
- [ ] All routes working
- [ ] Static files loading
- [ ] Model predictions working
- [ ] File upload functional

## 🐛 Troubleshooting

### Common Issues:

#### 1. 404 Error (Page Not Found)
- **Cause**: Routes not properly configured
- **Fix**: Check `wsgi.py` and `app.py` imports

#### 2. Static Files Not Loading
- **Cause**: Static folder path incorrect
- **Fix**: Update Flask static folder configuration

#### 3. Model Not Loading
- **Cause**: File paths incorrect in production
- **Fix**: Use absolute paths in `model_integration.py`

#### 4. Data Files Not Found
- **Cause**: Data directory path incorrect
- **Fix**: Update `data_dir` in `data_loader.py`

### Debug Commands:
```bash
# Check app logs
heroku logs --tail

# Check app status
heroku ps

# Restart app
heroku restart
```

## 🌟 Features After Deployment

### Working Features:
- ✅ Home page with statistics
- ✅ Model training interface
- ✅ File upload and analysis
- ✅ Real-time predictions
- ✅ Multiple ML algorithms
- ✅ NASA datasets integration

### API Endpoints:
- `GET /` - Home page
- `GET /train` - Training interface
- `GET /upload` - File upload
- `GET /predict` - Prediction interface
- `POST /train` - Train model
- `POST /upload` - Upload file
- `POST /predict_single` - Single prediction

## 📊 Performance Notes

### Memory Requirements:
- Minimum: 512MB RAM
- Recommended: 1GB RAM
- Model loading: ~100MB

### Startup Time:
- Initial deployment: 2-3 minutes
- Subsequent deployments: 1-2 minutes
- Cold start: 30-60 seconds

## 🔒 Security Considerations

### Production Settings:
- Debug mode disabled
- Secret key configured
- File upload limits set
- CORS properly configured

### Data Security:
- Uploaded files stored securely
- No sensitive data in logs
- Input validation implemented

---

## 📞 Support

If you encounter issues during deployment:
1. Check the logs for error messages
2. Verify all configuration files
3. Test locally first
4. Check platform-specific documentation

**Happy Deploying! 🚀**
