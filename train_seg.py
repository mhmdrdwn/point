def train(model, train_loader, val_loader,  epochs=8):
    for epoch in range(epochs):
        tick = time.time()
        print('Epoch ', epoch+1, ':')
        model.train()
        train_loss = 0.0
        for i, data in enumerate(tqdm(train_loader), 0):
            
            inputs, labels = data['points'].to(device), data['labels'].to(device)
            
            labels = labels.long()
            optimizer.zero_grad()
            outputs, m3x3, m64x64 = pointnet(inputs.transpose(1,2))

            loss = pointnetloss(outputs, labels, m3x3, m64x64)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        print('Train Loss: ', train_loss/len(train_loader))
        evaluate_per_epoch(model, train_loader, val_loader)
        tock = time.time()
        print('Time Elapsed: ', tock-tick, ' seconds')
        
    torch.save(model.state_dict(), "./final_model")
        
    
def evaluate_per_epoch(model, train_loader, val_loader):
            
    model.eval()
    train_correct = train_total = 0
    val_correct = val_total = 0
    train_ious, val_ious = [], []
        
    with torch.no_grad():
        for data in train_loader:
            inputs, labels = data['points'].to(device), data['labels'].to(device)
            outputs, __, __ = model(inputs.transpose(1,2))
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0) * labels.size(1) ##
            train_correct += (predicted == labels).sum().item()
            train_ious.append(intersection_over_union(outputs, labels))
    
    train_acc = train_correct / train_total
    
    print('Train mIOU: ', np.mean(train_ious))
    print('Train Accuracy: ',train_acc)
            
    # validation
    with torch.no_grad():
        for data in val_loader:
            inputs, labels = data['points'].to(device), data['labels'].to(device)
            outputs, __, __ = model(inputs.transpose(1,2))
            _, predicted = torch.max(outputs.data, 1)
            val_total += labels.size(0) * labels.size(1)
            val_correct += (predicted == labels).sum().item()
            val_ious.append(intersection_over_union(outputs, labels))
    val_acc = val_correct / val_total
    print('Valid mIOU: ', np.mean(val_ious))
    print('Valid Accuracy: ', val_acc)

