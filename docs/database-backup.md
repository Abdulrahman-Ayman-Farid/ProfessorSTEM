# Database Backup & Recovery Strategy

## Automated Nightly Backups

Supabase provides automated daily backups with 7-day retention. For additional safety:

1. **Automated Snapshots**: Supabase creates automatic snapshots every 24 hours
2. **Point-in-Time Recovery (PITR)**: Available for 7 days on Pro plans
3. **Manual Backups**: Run `pg_dump` weekly for off-site storage

## Manual Backup Procedure

```bash
# Create a full backup
pg_dump "$POSTGRES_URL" > backup_$(date +%Y%m%d).sql

# Compress the backup
gzip backup_$(date +%Y%m%d).sql

# Upload to S3
aws s3 cp backup_$(date +%Y%m%d).sql.gz s3://your-bucket/backups/
```

## Recovery Procedure

### Restore from Backup

```bash
# Create a new database
createdb professorstem_restore

# Restore the backup
psql "$POSTGRES_URL" -d professorstem_restore < backup_20240101.sql.gz
```

### Point-in-Time Recovery

1. Go to Supabase Dashboard → Project → Database → Backups
2. Select "Point-in-Time Recovery"
3. Choose the timestamp to restore to
4. Confirm the recovery (this creates a new database)

## Monitoring

- Set up alerts for backup failures
- Monitor storage usage
- Test restore procedures quarterly
